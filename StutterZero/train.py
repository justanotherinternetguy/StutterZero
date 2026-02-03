import os
import time
import yaml
import jiwer
import random
import shutil
import argparse
import datetime
import editdistance
import scipy.signal
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from models.encoder import Encoder
from models.decoder import Decoder
from models.asr_decoder import ASR_Decoder
from models.model import EEModel
from models.eval_distance import eval_wer, eval_cer
from models.data_loader import SpectrogramDataset, AudioDataLoader, AttrDict
import gc
from models.loss_function import Loss
from scipy.signal.windows import hamming

def load_label(label_path):
    char2index = dict() 
    index2char = dict() 
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#':
                continue
            index, char = line.split('   ')
            char = char.strip()
            if len(char) == 0:
                char = ' '
            char2index[char] = int(index)
            index2char[int(index)] = char
    return char2index, index2char

char2index, index2char = load_label('/home/alien/Git/StutterZero-Git/EEModel/label,csv/english_unit.labels')
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']

# def compute_cer(preds, labels):
#     total_wer = 0
#     total_cer = 0
#     total_wer_len = 0
#     total_cer_len = 0
#     for label, pred in zip(labels, preds):
#         units = []
#         units_pred = []
#         for a in label:
#             if a == EOS_token: 
#                 break
#             units.append(index2char[a])
#         for b in pred:
#             if b == EOS_token: 
#                 break
#             units_pred.append(index2char[b])
#         label = ''.join(units)
#         pred = ''.join(units_pred)
#         wer = eval_wer(pred, label)
#         cer = eval_cer(pred, label)
#         wer_len = len(label.split())
#         cer_len = len(label.replace(" ", ""))
#         total_wer += wer
#         total_cer += cer
#         total_wer_len += wer_len
#         total_cer_len += cer_len
#     return total_wer, total_cer, total_wer_len, total_cer_len

def compute_cer(preds, labels):
    total_wer = 0
    total_cer = 0
    total_wer_len = 0
    total_cer_len = 0
    for label, pred in zip(labels, preds):
        units = []
        units_pred = []
        for a in label:
            if a == EOS_token: 
                break
            units.append(index2char[a])
        for b in pred:
            if b == EOS_token: 
                break
            units_pred.append(index2char[b])
        label = ''.join(units)
        pred = ''.join(units_pred)
        
        # Calculate WER using jiwer
        wer = jiwer.wer(label, pred)  # Using jiwer to calculate WER
        cer = eval_cer(pred, label)  # Keep using your existing CER function
        
        wer_len = len(label.split())
        cer_len = len(label.replace(" ", ""))
        total_wer += wer
        total_cer += cer
        total_wer_len += wer_len
        total_cer_len += cer_len
    
    return total_wer, total_cer, total_wer_len, total_cer_len


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_num = 0
    total_wer = 0
    total_wer_len = 0
    start_time = time.time()
    total_batch_num = len(train_loader)
    
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        print(data)
        print(len(data))
        # seqs, targets, tts_seqs, seq_lengths, target_lengths, tts_seq_lengths = data
        seqs, targets, tts_seqs, seq_lengths, target_lengths, tts_seq_lengths, _ = data
        
        # Debug print statements (uncomment for debugging)
        print(f"Batch {i+1}/{total_batch_num}:")
        print(f"  seqs shape: {seqs.shape}")
        print(f"  targets shape: {targets.shape}")
        print(f"  tts_seqs shape: {tts_seqs.shape}")
        print(f"  Sequence lengths: {seq_lengths}")
        
        seqs = seqs.to(device)
        targets = targets.to(device)
        tts_seqs = tts_seqs.to(device)
        
        try:
            mel_outputs_postnet, mel_outputs, txt_outputs = model(seqs, tts_seqs, targets)
            loss = criterion(mel_outputs_postnet, mel_outputs, txt_outputs, targets, tts_seqs)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # Log learning rate after the optimizer step
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                print(f"Batch {i+1}/{total_batch_num} - Learning Rate: {lr}")
            
            hypothesis = txt_outputs.max(-1)[1]
            wer, _, wer_len, _ = compute_cer(hypothesis.cpu().numpy(), targets.cpu().numpy())
            total_wer += wer
            total_wer_len += wer_len
            
            if i % 100 == 0:
                print('{} train_batch: {:4d}/{:4d}, train_spec_loss: {:.4f}, train_wer: {:.4f}, train_time: {:.2f}'
                      .format(datetime.datetime.now(), i, total_batch_num, loss.item(), (wer / wer_len), time.time() - start_time))
            start_time = time.time()
            
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"Error occurred in batch {i}")
            print(f"Targets tensor: {targets}")
            raise e
    
    # Ensure you return the average loss and WER
    avg_loss = total_loss / total_batch_num
    avg_wer = total_wer / total_wer_len if total_wer_len > 0 else 0  # Avoid division by zero
    
    return avg_loss, avg_wer


def evaluation(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_num = 0
    total_wer = 0
    total_wer_len = 0
    start_time = time.time()
    total_batch_num = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            seqs, targets, tts_seqs, seq_lengths, target_lengths, tts_seq_lengths = data
            seqs = seqs.to(device) 
            targets = targets.to(device)
            tts_seqs = tts_seqs.to(device)
            mel_outputs_postnet, mel_outputs, txt_outputs = model(seqs, tts_seqs, targets)
            loss = criterion(mel_outputs_postnet, mel_outputs, txt_outputs, targets, tts_seqs)
            total_loss += loss.item()
            hypothesis = txt_outputs.max(-1)[1]
            wer, _, wer_len, _ = compute_cer(hypothesis.cpu().numpy(), targets.cpu().numpy())
            total_wer += wer
            total_wer_len += wer_len
    eval_loss = total_loss / total_batch_num
    final_wer = (total_wer / total_wer_len) * 100
    return eval_loss, final_wer

def main():
    yaml_name = "/home/alien/Git/EEModel/label,csv/config.yaml"
    with open("./ee.txt", "w") as f:
        f.write(yaml_name)
        f.write('\n')
        f.write('\n')
    
    configfile = open(yaml_name)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    
    random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)
    
    # Use specific GPU (change index as needed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    windows = { 'hamming': hamming,
               'hann': scipy.signal.windows.hann,
               'blackman': scipy.signal.windows.blackman,
               'bartlett': scipy.signal.windows.bartlett }
    
    SAMPLE_RATE = config.audio_data.sampling_rate
    WINDOW_SIZE = config.audio_data.window_size
    WINDOW_STRIDE = config.audio_data.window_stride
    WINDOW = config.audio_data.window
    
    audio_conf = dict(sample_rate=SAMPLE_RATE,
                     window_size=WINDOW_SIZE,
                     window_stride=WINDOW_STRIDE,
                     window=WINDOW)
    
    hop_length = int(round(SAMPLE_RATE * 0.001 * WINDOW_STRIDE))

    #-------------------------- Model Initialize --------------------------
    enc = Encoder(rnn_hidden_size=256,
                 dropout=0.5,
                 bidirectional=True)
    dec = Decoder(target_dim=1025,
                  pre_net_dim=256,
                  rnn_hidden_size=1024,
                  encoder_dim=256*2,
                  attention_dim=128,
                  attention_filter_n=32,
                  attention_filter_len=31,
                  postnet_hidden_size=512,
                  postnet_filter=5,
                  dropout=0.5)
    asr_dec = ASR_Decoder(label_dim=31,
                         embedding_dim=64,
                         encoder_dim=256*2,
                         rnn_hidden_size=512,
                         second_rnn_hidden_size=256,
                         attention_dim=128,
                         attention_filter_n=32,
                         attention_filter_len=31,
                         sos_id=SOS_token,
                         eos_id=EOS_token,
                         pad_id=PAD_token)
    
    # Create model and move to specified GPU
    model = EEModel(enc, dec, asr_dec).to(device)
    
    #-------------------------- Loss Initialize --------------------------
    spec_criterion = nn.MSELoss()
    asr_criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = Loss(spec_criterion, asr_criterion)
    
    #-------------------- Model Optimizer --------------------
    optimizer = optim.Adam(model.parameters(),  # Removed .module
                          lr=config.optim.lr,
                          betas=(0.9, 0.999),
                          eps=1e-06,
                          weight_decay=1e-06)
    
    #-------------------------- Data load --------------------------
    train_dataset = SpectrogramDataset(audio_conf,
                                      "/home/alien/Git/StutterZero-Git/EEModel/label,csv/train.csv",
                                      feature_type=config.audio_data.type,
                                      normalize=True,
                                      spec_augment=True)
    
    # Set num_workers=0 for single GPU training
    train_loader = AudioDataLoader(dataset=train_dataset,
                                  shuffle=False,
                                  num_workers=0,
                                  batch_size=3,
                                  drop_last=True)
    
    val_dataset = SpectrogramDataset(audio_conf,
                                    "/home/alien/Git/StutterZero-Git/EEModel/label,csv/test.csv",
                                    feature_type=config.audio_data.type,
                                    normalize=True,
                                    spec_augment=False)
    
    val_loader = AudioDataLoader(dataset=val_dataset,
                                shuffle=False,
                                num_workers=0,
                                batch_size=3,
                                drop_last=True)

    print("\nTraining on device:", device)
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")

    print(" ")

    pre_test_loss = 100000
    
    for epoch in range(config.training.begin_epoch, config.training.end_epoch):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            print("lr = ", lr)
        
        print('{} train'.format(datetime.datetime.now()))
        train_time = time.time()
        train_loss, train_wer = train(model, train_loader, optimizer, criterion, device)
        train_total_time = time.time() - train_time
        
        print('{} Epoch {} (Train) Loss {:.4f}, WER {:.2f}, time: {:.2f}'.format(
              datetime.datetime.now(), epoch+1, train_loss, train_wer, train_total_time))
        
        print('{} eval'.format(datetime.datetime.now()))
        eval_time = time.time()
        eval_loss, test_wer = evaluation(model, val_loader, criterion, device)
        eval_total_time = time.time() - eval_time
        
        print('{} Epoch {} (Eval) Loss {:.4f}, WER {:.2f}, time: {:.2f}'.format(
              datetime.datetime.now(), epoch+1, eval_loss, test_wer, eval_total_time))
        
        with open("./ee_yes_asr.txt", "a") as f:
            f.write("lr = " + str(lr))
            f.write('\n')
            f.write('Epoch %d (Train) Loss %0.4f WER %0.4f time %0.4f' % (
                    epoch+1, train_loss, train_wer, train_total_time))
            f.write('\n')
            f.write('Epoch %d (Eval) Loss %0.4f WER %0.4f time %0.4f' % (
                    epoch+1, eval_loss, test_wer, eval_total_time))
            f.write('\n')
        
        if pre_test_loss > eval_loss:
            torch.save(model.state_dict(), "./plz_load/best_ee.pth")
            pre_test_loss = eval_loss
        
        torch.save(model.state_dict(), "./plz_load/ee.pth")

if __name__ == '__main__':
    main()
