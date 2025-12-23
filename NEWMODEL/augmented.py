import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.amp import autocast, GradScaler
from typing import Tuple, Optional, Dict, Any
import gc
import math
import librosa
from torch.nn.utils.rnn import pad_sequence
import random

class Config:
    """Model configuration"""

    sample_rate = 16000
    n_mels = 128  

    n_fft = 1024  

    hop_length = 256  

    win_length = 1024

    encoder_dim = 256  

    decoder_dim = 256  

    attention_dim = 128  

    num_encoder_layers = 3  

    num_decoder_layers = 3  

    num_heads = 8
    dropout = 0.3  

    vocab_size = 31  

    pad_token = 0
    sos_token = 1
    eos_token = 2

    max_audio_length = 800  

    max_text_length = 200   

    label_smoothing = 0.1
    mixup_alpha = 0.2
    spec_augment = True

    freq_mask_num = 2  

    freq_mask_width = 15  

    time_mask_num = 2  

    time_mask_width = 30  

    time_warp_w = 10  

    spec_augment_prob = 0.5  

class SpecAugment(nn.Module):
    """SpecAugment implementation for mel spectrograms"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def time_warp(self, mel_spec, W=5):
        """Apply time warping to mel spectrogram
        Args:
            mel_spec: [time, freq] mel spectrogram
            W: time warp parameter
        """
        if mel_spec.size(0) <= 1:
            return mel_spec

        time_steps = mel_spec.size(0)

        if time_steps - W <= W:
            return mel_spec

        center = random.randint(W, time_steps - W)
        warped = random.randint(-W, W)

        left = torch.linspace(0, center - 1, center)
        right = torch.linspace(center + warped, time_steps - 1, time_steps - center)

        if warped >= 0:

            warped_indices = torch.cat([
                left,
                torch.linspace(center, center + warped, warped + 1),
                right + warped
            ])
        else:

            warped_indices = torch.cat([
                left * (center + warped) / center,
                right
            ])

        warped_indices = torch.clamp(warped_indices, 0, time_steps - 1)

        mel_spec_warped = F.interpolate(
            mel_spec.unsqueeze(0).unsqueeze(0),  

            size=(len(warped_indices), mel_spec.size(1)),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        return mel_spec_warped

    def frequency_mask(self, mel_spec, F=30, num_masks=1):
        """Apply frequency masking
        Args:
            mel_spec: [time, freq] mel spectrogram
            F: maximum frequency mask width
            num_masks: number of masks to apply
        """
        cloned = mel_spec.clone()
        num_mel_channels = cloned.shape[1]

        for _ in range(num_masks):
            if F >= num_mel_channels:
                continue

            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            cloned[:, f_zero:f_zero + f] = 0

        return cloned

    def time_mask(self, mel_spec, T=40, num_masks=1):
        """Apply time masking
        Args:
            mel_spec: [time, freq] mel spectrogram
            T: maximum time mask width
            num_masks: number of masks to apply
        """
        cloned = mel_spec.clone()
        len_spectro = cloned.shape[0]

        for _ in range(num_masks):
            if T >= len_spectro:
                continue

            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            cloned[t_zero:t_zero + t, :] = 0

        return cloned

    def forward(self, mel_spec):
        """Apply SpecAugment transformations
        Args:
            mel_spec: [time, freq] mel spectrogram
        Returns:
            augmented mel spectrogram
        """

        if self.config.time_warp_w > 0:
            mel_spec = self.time_warp(mel_spec, self.config.time_warp_w)

        if self.config.freq_mask_num > 0:
            mel_spec = self.frequency_mask(
                mel_spec, 
                self.config.freq_mask_width, 
                self.config.freq_mask_num
            )

        if self.config.time_mask_num > 0:
            mel_spec = self.time_mask(
                mel_spec, 
                self.config.time_mask_width, 
                self.config.time_mask_num
            )

        return mel_spec

class StutteredSpeechDataset(Dataset):
    """Dataset class for stuttered speech conversion with SpecAugment"""

    def __init__(self, csv_path: str, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.data = pd.read_csv(csv_path)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length
        )

        self.spec_augment = SpecAugment(config)

        self.char_to_idx = self._build_vocab()

        self._original_length = len(self.data)

    def _build_vocab(self):
        """Build restricted grapheme vocabulary"""
        allowed_chars = "abcdefghijklmnopqrstuvwxyz '"

        char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        for i, char in enumerate(allowed_chars, start=3):
            char_to_idx[char] = i

        return char_to_idx

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        try:
            waveform, sr = torchaudio.load(audio_path)

            if sr != self.config.sample_rate: #cpu
                resample = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resample(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            mel_spec = self.mel_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-8)  

            return mel_spec.squeeze(0).transpose(0, 1).to(self.device)

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return torch.zeros(1, self.config.n_mels, device=self.device)

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Convert text to token indices"""
        if not isinstance(text, str):
            return torch.tensor([self.config.sos_token, self.config.eos_token], device=self.device)

        tokens = [self.config.sos_token]
        for char in text.lower():
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
        tokens.append(self.config.eos_token)

        return torch.tensor(tokens, device=self.device)

    def __len__(self):

        return self._original_length * 2

    def __getitem__(self, idx: int) -> dict:

        is_augmented = idx >= self._original_length
        original_idx = idx % self._original_length

        row = self.data.iloc[original_idx]

        source_audio = self._load_audio(row.iloc[0])

        transcript_path = row.iloc[1].strip()
        with open(transcript_path, "r") as f:
            transcript_text = f.read().strip()

        transcript = self._tokenize_text(transcript_text)
        target_audio = self._load_audio(row.iloc[2])

        if is_augmented and self.config.spec_augment:

            if random.random() < self.config.spec_augment_prob:
                source_audio = self.spec_augment(source_audio)
            if random.random() < self.config.spec_augment_prob:
                target_audio = self.spec_augment(target_audio)

        return {
            "input_audio": source_audio,
            "transcript": transcript,
            "target_audio": target_audio,
            "is_augmented": is_augmented
        }

def collate_fn(batch, config):
    """Collate function for DataLoader"""
    input_audios = [item['input_audio'] for item in batch]
    target_audios = [item['target_audio'] for item in batch]
    transcripts = [item['transcript'] for item in batch]
    is_augmented = [item['is_augmented'] for item in batch]

    input_audios = pad_sequence(input_audios, batch_first=True, padding_value=0)
    target_audios = pad_sequence(target_audios, batch_first=True, padding_value=0)
    transcripts = pad_sequence(transcripts, batch_first=True, padding_value=config.pad_token)

    return {
        'input_audio': input_audios,
        'target_audio': target_audios,
        'transcript': transcripts,
        'is_augmented': is_augmented
    }

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        return self.w_o(context)

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x

class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, self_mask=None, cross_mask=None):

        self_attn_out = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out = self.cross_attn(x, encoder_out, encoder_out, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x

class SpeechEncoder(nn.Module):
    """Shared speech encoder using transformer architecture"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.n_mels, config.encoder_dim)

        self.pos_encoding = PositionalEncoding(config.encoder_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.encoder_dim, 
                config.num_heads,
                config.encoder_dim * 4,
                config.dropout
            ) for _ in range(config.num_encoder_layers)
        ])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):

        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x

class SpectrogramDecoder(nn.Module):
    """Decoder for generating mel spectrograms"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.prenet = nn.Sequential(
            nn.Linear(config.encoder_dim, config.decoder_dim),
            nn.BatchNorm1d(config.decoder_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_dim, config.decoder_dim),
            nn.BatchNorm1d(config.decoder_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.attention_layers = nn.ModuleList([
            TransformerDecoderLayer(
                config.decoder_dim,
                config.num_heads,
                config.decoder_dim * 4,
                config.dropout
            ) for _ in range(3)  

        ])

        self.postnet = nn.Sequential(
            nn.Conv1d(config.n_mels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(config.dropout),

            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(config.dropout),

            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(config.dropout),

            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(config.dropout),

            nn.Conv1d(512, config.n_mels, kernel_size=5, padding=2)
        )

        self.mel_proj_coarse = nn.Linear(config.decoder_dim, config.n_mels)
        self.mel_proj_fine = nn.Linear(config.decoder_dim, config.n_mels)

        self.stop_proj = nn.Linear(config.decoder_dim, 1)

        self.dropout = nn.Dropout(config.dropout)

    def apply_prenet(self, x):
        """Apply prenet with proper batch norm handling"""
        batch_size, seq_len, dim = x.shape

        x_reshaped = x.reshape(-1, dim)

        x_reshaped = self.prenet[0](x_reshaped)  

        x_reshaped = self.prenet[1](x_reshaped)  

        x_reshaped = self.prenet[2](x_reshaped)  

        x_reshaped = self.prenet[3](x_reshaped)  

        x_reshaped = self.prenet[4](x_reshaped)  

        x_reshaped = self.prenet[5](x_reshaped)  

        x_reshaped = self.prenet[6](x_reshaped)  

        x_reshaped = self.prenet[7](x_reshaped)  

        return x_reshaped.reshape(batch_size, seq_len, -1)

    def forward(self, encoder_out, target=None, mask=None):
        batch_size, enc_len, _ = encoder_out.shape

        if target is not None:
            target_len = target.size(1)
        else:
            target_len = enc_len

        if target_len != enc_len:
            encoder_out = F.interpolate(
                encoder_out.transpose(1, 2), 
                size=target_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)

        x = self.apply_prenet(encoder_out)

        for layer in self.attention_layers:
            residual = x
            x = layer(x, encoder_out)
            x = x + residual  

        mel_coarse = self.mel_proj_coarse(x)
        mel_fine = self.mel_proj_fine(x)
        mel_output = mel_coarse + 0.3 * mel_fine  

        stop_tokens = torch.sigmoid(self.stop_proj(x))

        mel_post_input = mel_output.transpose(1, 2)  

        mel_post_residual = self.postnet(mel_post_input)

        mel_output_refined = mel_post_input + mel_post_residual
        mel_output_refined = mel_output_refined.transpose(1, 2)  

        return {
            'mel_output': mel_output_refined,
            'stop_tokens': stop_tokens,
            'mel_coarse': mel_coarse,
            'mel_fine': mel_fine
        }

class TranscriptDecoder(nn.Module):
    """Decoder for generating text transcripts"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.decoder_dim)

        self.pos_encoding = PositionalEncoding(config.decoder_dim)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                config.decoder_dim,
                config.num_heads,
                config.decoder_dim * 4,
                config.dropout
            ) for _ in range(config.num_decoder_layers)
        ])

        self.output_proj = nn.Linear(config.decoder_dim, config.vocab_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, encoder_out, target=None, mask=None):
        if target is not None:

            target_embed = self.embedding(target)
            target_embed = self.pos_encoding(target_embed)
            target_embed = self.dropout(target_embed)

            for layer in self.layers:
                target_embed = layer(target_embed, encoder_out, mask)

            output = self.output_proj(target_embed)
            return output
        else:

            batch_size = encoder_out.size(0)
            max_len = self.config.max_text_length

            outputs = []
            current_tokens = torch.full((batch_size, 1), self.config.sos_token).to(encoder_out.device)

            for _ in range(max_len):
                token_embed = self.embedding(current_tokens)
                token_embed = self.pos_encoding(token_embed)

                for layer in self.layers:
                    token_embed = layer(token_embed, encoder_out)

                output = self.output_proj(token_embed[:, -1:, :])
                next_token = output.argmax(dim=-1)

                outputs.append(output)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

                if (next_token == self.config.eos_token).all():
                    break

            return torch.cat(outputs, dim=1)

class MultitaskStutteredSpeechModel(nn.Module):
    """Main multitask model for stuttered speech conversion"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.encoder = SpeechEncoder(config)

        self.spectrogram_decoder = SpectrogramDecoder(config)
        self.transcript_decoder = TranscriptDecoder(config)

    def forward(self, input_audio, target_audio=None, transcript=None):
        """
        Args:
            input_audio: [batch_size, seq_len, n_mels] - stuttered speech mel spectrogram
            target_audio: [batch_size, seq_len, n_mels] - target fluent speech mel spectrogram
            transcript: [batch_size, seq_len] - target transcript tokens
        """

        encoder_out = self.encoder(input_audio)

        spec_output_dict = self.spectrogram_decoder(encoder_out, target_audio)

        if transcript is not None:

            transcript_input = transcript[:, :-1]
            transcript_output = self.transcript_decoder(encoder_out, transcript_input)
        else:
            transcript_output = self.transcript_decoder(encoder_out)

        return {
            'spectrogram': spec_output_dict,  

            'transcript': transcript_output
        }

class StutteredSpeechLoss(nn.Module):
    """Combined loss for multitask learning"""

    def __init__(self, spec_weight=1.0, transcript_weight=0.1, l1_weight=0.5, 
                 stop_weight=0.1, spectral_weight=0.3):
        super().__init__()
        self.spec_weight = spec_weight
        self.transcript_weight = transcript_weight
        self.l1_weight = l1_weight
        self.stop_weight = stop_weight
        self.spectral_weight = spectral_weight

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()  

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.transcript_loss = nn.CrossEntropyLoss(ignore_index=0)

    def spectral_convergence_loss(self, pred, target):
        """Spectral convergence loss for better frequency domain modeling"""
        pred = pred.float()
        target = target.float()
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return torch.norm(pred_mag - target_mag, p='fro') / torch.norm(target_mag, p='fro')

    def forward(self, predictions, targets):

        spec_output = predictions['spectrogram']['mel_output']
        target_audio = targets['target_audio']

        mse_loss = self.mse_loss(spec_output, target_audio)
        l1_loss = self.l1_loss(spec_output, target_audio)

        spectral_loss = self.spectral_convergence_loss(spec_output, target_audio)

        stop_tokens = predictions['spectrogram']['stop_tokens']
        batch_size, seq_len = stop_tokens.shape[:2]

        stop_targets = torch.zeros_like(stop_tokens.squeeze(-1))
        stop_targets[:, -1] = 1.0  

        stop_loss = self.bce_loss(stop_tokens.squeeze(-1), stop_targets)

        transcript_pred = predictions['transcript'].reshape(-1, predictions['transcript'].size(-1))
        transcript_target = targets['transcript'][:, 1:].reshape(-1)
        transcript_loss = self.transcript_loss(transcript_pred, transcript_target)

        coarse_loss = self.mse_loss(predictions['spectrogram']['mel_coarse'], target_audio)
        fine_loss = self.l1_loss(predictions['spectrogram']['mel_fine'], target_audio)

        total_loss = (self.spec_weight * mse_loss + 
                     self.l1_weight * l1_loss +
                     self.spectral_weight * spectral_loss +
                     self.stop_weight * stop_loss +
                     self.transcript_weight * transcript_loss +
                     0.3 * coarse_loss +
                     0.2 * fine_loss)

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'l1_loss': l1_loss,
            'spectral_loss': spectral_loss,
            'stop_loss': stop_loss,
            'transcript_loss': transcript_loss,
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss
        }

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    config = Config()

    dataset = StutteredSpeechDataset(
        '/home/alien/Git/StutterZero/EEModel/label,csv/train2.csv', 
        config, 
        device
    )

    print(f"Dataset size (with augmentation): {len(dataset)}")
    print(f"Original dataset size: {dataset._original_length}")

    dataloader = DataLoader(
        dataset, 
        batch_size=2,
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, config)
    )

    model = MultitaskStutteredSpeechModel(config).to(device)
    criterion = StutteredSpeechLoss(spec_weight=1.0, transcript_weight=0.5)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        betas=(0.9, 0.98), 
        weight_decay=1e-5, 
        eps=1e-6
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50, 
        T_mult=2, 
        eta_min=1e-6
    )

    scaler = GradScaler('cuda')

    model.train()
    best_loss = float('inf')

    augmented_losses = []
    original_losses = []

    for epoch in range(3500):
        total_loss = 0
        epoch_augmented_losses = []
        epoch_original_losses = []

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(
                    batch['input_audio'],
                    batch['target_audio'],
                    batch['transcript']
                )
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['total_loss']

                for i, is_aug in enumerate(batch['is_augmented']):
                    sample_loss = loss.item()  

                    if is_aug:
                        epoch_augmented_losses.append(sample_loss)
                    else:
                        epoch_original_losses.append(sample_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

                aug_count = sum(batch['is_augmented'])
                orig_count = len(batch['is_augmented']) - aug_count

                print(f'Epoch {epoch}, Batch {batch_idx}, Total Loss: {loss.item():.6f}')
                print(f'  Batch composition: {orig_count} original, {aug_count} augmented samples')

                print(f'  MSE: {loss_dict["mse_loss"].item():.6f}, '
                      f'L1: {loss_dict["l1_loss"].item():.6f}, '
                      f'Spectral: {loss_dict["spectral_loss"].item():.6f}')

        avg_loss = total_loss / len(dataloader)

        if epoch_augmented_losses:
            avg_aug_loss = np.mean(epoch_augmented_losses)
            augmented_losses.append(avg_aug_loss)
        if epoch_original_losses:
            avg_orig_loss = np.mean(epoch_original_losses)
            original_losses.append(avg_orig_loss)

        print(f'Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}')
        if epoch_augmented_losses and epoch_original_losses:
            print(f'  Avg Augmented Loss: {avg_aug_loss:.6f}')
            print(f'  Avg Original Loss: {avg_orig_loss:.6f}')

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_stuttered_speech_model.pth')
            print(f"New best model saved with loss: {best_loss:.6f}")

    torch.save(model.state_dict(), 'final_stuttered_speech_model.pth')
    print("Training completed!")

    if augmented_losses and original_losses:
        print(f"\nFinal Training Statistics:")
        print(f"Average loss on original samples: {np.mean(original_losses):.6f}")
        print(f"Average loss on augmented samples: {np.mean(augmented_losses):.6f}")
        print(f"Loss difference (aug - orig): {np.mean(augmented_losses) - np.mean(original_losses):.6f}")

if __name__ == "__main__":
    train_model()