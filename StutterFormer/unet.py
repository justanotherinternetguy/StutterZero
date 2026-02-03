import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.amp import autocast, GradScaler
from torchsummary import summary
from typing import Tuple, Optional, Dict, Any, List
import gc
import math
import matplotlib.pyplot as plt
import librosa
from torch.nn.utils.rnn import pad_sequence

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
    num_encoder_layers = 6  

    num_decoder_layers = 6  

    num_heads = 4
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

class StutteredSpeechDataset(Dataset):
    """Dataset class for stuttered speech conversion"""

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

        self.char_to_idx = self._build_vocab()

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

            if sr != self.config.sample_rate:
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
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        source_audio = self._load_audio(row.iloc[0])

        transcript_path = row.iloc[1].strip()
        with open(transcript_path, "r") as f:
            transcript_text = f.read().strip()

        transcript = self._tokenize_text(transcript_text)
        target_audio = self._load_audio(row.iloc[2])

        return {
            "input_audio": source_audio,
            "transcript": transcript,
            "target_audio": target_audio,
        }

def collate_fn(batch, config):
    """Collate function for DataLoader"""
    input_audios = [item['input_audio'] for item in batch]
    target_audios = [item['target_audio'] for item in batch]
    transcripts = [item['transcript'] for item in batch]

    input_audios = pad_sequence(input_audios, batch_first=True, padding_value=0)
    target_audios = pad_sequence(target_audios, batch_first=True, padding_value=0)
    transcripts = pad_sequence(transcripts, batch_first=True, padding_value=config.pad_token)

    return {
        'input_audio': input_audios,
        'target_audio': target_audios,
        'transcript': transcripts
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
    """Transformer decoder layer with skip connection support"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, skip_dim: Optional[int] = None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        self.skip_proj = None
        if skip_dim is not None and skip_dim != d_model:
            self.skip_proj = nn.Linear(skip_dim, d_model)

        ff_input_dim = d_model * 2 if skip_dim is not None else d_model
        self.feed_forward = nn.Sequential(
            nn.Linear(ff_input_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, self_mask=None, cross_mask=None, skip_connection=None):

        self_attn_out = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out = self.cross_attn(x, encoder_out, encoder_out, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        if skip_connection is not None:

            if skip_connection.size(1) != x.size(1):
                skip_connection = F.interpolate(
                    skip_connection.transpose(1, 2),
                    size=x.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            if self.skip_proj is not None:
                skip_connection = self.skip_proj(skip_connection)

            x_with_skip = torch.cat([x, skip_connection], dim=-1)
        else:
            x_with_skip = x

        ff_out = self.feed_forward(x_with_skip)
        x = self.norm3(x + self.dropout(ff_out))

        return x

class UNetSpeechEncoder(nn.Module):
    """U-Net style encoder that returns intermediate features for skip connections"""

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

        skip_connections = []

        for layer in self.layers:
            x = layer(x, mask)
            skip_connections.append(x)  

        return x, skip_connections

class UNetSpectrogramDecoder(nn.Module):
    """U-Net style decoder with skip connections from encoder"""

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
                config.dropout,
                skip_dim=config.encoder_dim  

            ) for _ in range(config.num_decoder_layers)
        ])

        self.skip_fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.encoder_dim * 2),
                nn.Linear(config.encoder_dim * 2, config.encoder_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            ) for _ in range(config.num_decoder_layers)
        ])

        self.postnet = nn.Sequential(

            self._make_residual_block(config.n_mels, 512),
            self._make_residual_block(512, 512),
            self._make_residual_block(512, 512),

            nn.Conv1d(512, config.n_mels, kernel_size=1)
        )

        self.mel_proj_coarse = nn.Linear(config.decoder_dim, config.n_mels)
        self.mel_proj_fine = nn.Linear(config.decoder_dim, config.n_mels)
        self.stop_proj = nn.Linear(config.decoder_dim, 1)

        self.attention_gates = nn.ModuleList([
            self._make_attention_gate(config.encoder_dim, config.decoder_dim)
            for _ in range(config.num_decoder_layers)
        ])

        self.dropout = nn.Dropout(config.dropout)

    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block for the postnet"""
        layers = []

        layers.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        ])

        layers.extend([
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels)
        ])

        main_path = nn.Sequential(*layers)

        if in_channels != out_channels:
            shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            shortcut = nn.Identity()

        return ResidualBlock(main_path, shortcut)

    def _make_attention_gate(self, encoder_dim, decoder_dim):
        """Create attention gate for skip connections"""
        return nn.Sequential(
            nn.Linear(encoder_dim + decoder_dim, encoder_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_dim // 2, encoder_dim),
            nn.Sigmoid()
        )

    def apply_prenet(self, x):
        """Apply prenet with proper batch norm handling"""
        batch_size, seq_len, dim = x.shape
        x_reshaped = x.reshape(-1, dim)

        for i in range(0, len(self.prenet), 4):  

            x_reshaped = self.prenet[i](x_reshaped)      

            x_reshaped = self.prenet[i+1](x_reshaped)    

            x_reshaped = self.prenet[i+2](x_reshaped)    

            x_reshaped = self.prenet[i+3](x_reshaped)    

        return x_reshaped.reshape(batch_size, seq_len, -1)

    def forward(self, encoder_out, skip_connections, target=None, mask=None):
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

        reversed_skip_connections = list(reversed(skip_connections))

        for i, (layer, skip_feat) in enumerate(zip(self.attention_layers, reversed_skip_connections)):

            if skip_feat.size(1) != x.size(1):
                skip_feat = F.interpolate(
                    skip_feat.transpose(1, 2),
                    size=x.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            attention_input = torch.cat([skip_feat, x], dim=-1)
            attention_weights = self.attention_gates[i](attention_input)
            skip_feat_gated = skip_feat * attention_weights

            residual = x
            x = layer(x, encoder_out, skip_connection=skip_feat_gated)
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

class ResidualBlock(nn.Module):
    """Proper residual block implementation"""
    def __init__(self, main_path, shortcut):
        super().__init__()
        self.main_path = main_path
        self.shortcut = shortcut

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main_path(x)
        return F.relu(out + residual)

class ResidualConnection(nn.Module):
    """Residual connection module - deprecated, use ResidualBlock instead"""
    def __init__(self, shortcut):
        super().__init__()
        self.shortcut = shortcut

    def forward(self, x):
        residual = self.shortcut(x)
        return F.relu(x + residual)

class TranscriptDecoder(nn.Module):
    """Decoder for generating text transcripts with skip connections"""

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
                config.dropout,
                skip_dim=config.encoder_dim
            ) for _ in range(config.num_decoder_layers)
        ])

        self.output_proj = nn.Linear(config.decoder_dim, config.vocab_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, encoder_out, skip_connections, target=None, mask=None):
        if target is not None:

            target_embed = self.embedding(target)
            target_embed = self.pos_encoding(target_embed)
            target_embed = self.dropout(target_embed)

            reversed_skip_connections = list(reversed(skip_connections))

            for layer, skip_feat in zip(self.layers, reversed_skip_connections):
                target_embed = layer(target_embed, encoder_out, mask, skip_connection=skip_feat)

            output = self.output_proj(target_embed)
            return output
        else:

            batch_size = encoder_out.size(0)
            max_len = self.config.max_text_length

            reversed_skip_connections = list(reversed(skip_connections))

            outputs = []
            current_tokens = torch.full((batch_size, 1), self.config.sos_token).to(encoder_out.device)

            for _ in range(max_len):
                token_embed = self.embedding(current_tokens)
                token_embed = self.pos_encoding(token_embed)

                for layer, skip_feat in zip(self.layers, reversed_skip_connections):
                    token_embed = layer(token_embed, encoder_out, skip_connection=skip_feat)

                output = self.output_proj(token_embed[:, -1:, :])
                next_token = output.argmax(dim=-1)

                outputs.append(output)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

                if (next_token == self.config.eos_token).all():
                    break

            return torch.cat(outputs, dim=1)

class UNetMultitaskStutteredSpeechModel(nn.Module):
    """U-Net style multitask model with skip connections"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.encoder = UNetSpeechEncoder(config)

        self.spectrogram_decoder = UNetSpectrogramDecoder(config)
        self.transcript_decoder = TranscriptDecoder(config)

    def forward(self, input_audio, target_audio=None, transcript=None):
        """
        Args:
            input_audio: [batch_size, seq_len, n_mels] - stuttered speech mel spectrogram
            target_audio: [batch_size, seq_len, n_mels] - target fluent speech mel spectrogram
            transcript: [batch_size, seq_len] - target transcript tokens
        """

        encoder_out, skip_connections = self.encoder(input_audio)

        spec_output_dict = self.spectrogram_decoder(encoder_out, skip_connections, target_audio)

        if transcript is not None:

            transcript_input = transcript[:, :-1]
            transcript_output = self.transcript_decoder(encoder_out, skip_connections, transcript_input)
        else:
            transcript_output = self.transcript_decoder(encoder_out, skip_connections)

        return {
            'spectrogram': spec_output_dict,
            'transcript': transcript_output
        }

class StutteredSpeechLoss(nn.Module):
    """Combined loss for multitask learning with U-Net"""

    def __init__(self, spec_weight=1.0, transcript_weight=0.1, l1_weight=0.5, 
                 stop_weight=0.1, spectral_weight=0.3, perceptual_weight=0.2):
        super().__init__()
        self.spec_weight = spec_weight
        self.transcript_weight = transcript_weight
        self.l1_weight = l1_weight
        self.stop_weight = stop_weight
        self.spectral_weight = spectral_weight
        self.perceptual_weight = perceptual_weight

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

    def perceptual_loss(self, pred, target):
        """Simple perceptual loss using gradient differences"""

        pred_grad_time = torch.diff(pred, dim=1)
        target_grad_time = torch.diff(target, dim=1)

        pred_grad_freq = torch.diff(pred, dim=2)
        target_grad_freq = torch.diff(target, dim=2)

        time_loss = F.l1_loss(pred_grad_time, target_grad_time)
        freq_loss = F.l1_loss(pred_grad_freq, target_grad_freq)

        return time_loss + freq_loss

    def forward(self, predictions, targets):

        spec_output = predictions['spectrogram']['mel_output']
        target_audio = targets['target_audio']

        mse_loss = self.mse_loss(spec_output, target_audio)
        l1_loss = self.l1_loss(spec_output, target_audio)

        spectral_loss = self.spectral_convergence_loss(spec_output, target_audio)

        perceptual_loss = self.perceptual_loss(spec_output, target_audio)

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
                     self.perceptual_weight * perceptual_loss +
                     self.stop_weight * stop_loss +
                     self.transcript_weight * transcript_loss +
                     0.3 * coarse_loss +
                     0.2 * fine_loss)

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'l1_loss': l1_loss,
            'spectral_loss': spectral_loss,
            'perceptual_loss': perceptual_loss,
            'stop_loss': stop_loss,
            'transcript_loss': transcript_loss,
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss
        }

def train_unet_model():
    """Training function for U-Net model with skip connections"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    config = Config()
    writer = SummaryWriter(log_dir="./runs/unet_stuttered_speech_train")

    dataset = StutteredSpeechDataset(
        '/home/alien/Git/StutterZero/EEModel/label,csv/train2.csv', 
        config, 
        device
    )

    batch_size = 1  

    accumulation_steps = 8  

    effective_batch_size = batch_size * accumulation_steps

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, config)
    )

    model = UNetMultitaskStutteredSpeechModel(config).to(device)

    criterion = StutteredSpeechLoss(
        spec_weight=1.0, 
        transcript_weight=0.5, 
        perceptual_weight=0.2
    )

    base_lr = 1e-4
    lr = base_lr * math.sqrt(effective_batch_size / 8)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
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
    global_step = 0

    print(f"Training U-Net model with skip connections:")
    print(f"Physical batch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Adjusted learning rate: {lr}")
    print(f"Encoder layers: {config.num_encoder_layers}")
    print(f"Decoder layers: {config.num_decoder_layers}")

    for epoch in range(10000):
        total_loss = 0
        accumulated_loss = 0
        accumulated_loss_dict = {}

        for batch_idx, batch in enumerate(dataloader):

            with autocast('cuda'):
                outputs = model(
                    batch['input_audio'],
                    batch['target_audio'],
                    batch['transcript']
                )
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['total_loss']

                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            accumulated_loss += loss.item() * accumulation_steps

            for key, value in loss_dict.items():
                if key not in accumulated_loss_dict:
                    accumulated_loss_dict[key] = 0
                accumulated_loss_dict[key] += value.item()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

                num_accumulated = min(accumulation_steps, (batch_idx % accumulation_steps) + 1)
                avg_accumulated_loss = accumulated_loss / num_accumulated

                if global_step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                    print(f'Epoch {epoch}, Step {global_step}, Batch {batch_idx}, '
                          f'Loss: {avg_accumulated_loss:.6f}')

                    writer.add_scalar("Loss/total", avg_accumulated_loss, global_step)

                    for key, value in accumulated_loss_dict.items():
                        if key != 'total_loss':
                            writer.add_scalar(f"Loss/{key}", value / num_accumulated, global_step)

                    writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)

                    if batch['target_audio'].size(0) > 0:
                        writer.add_image("Mel/target", 
                                        batch['target_audio'][0].T.cpu().numpy(), 
                                        global_step, dataformats="HW")
                        writer.add_image("Mel/predicted", 
                                        outputs["spectrogram"]["mel_output"][0].T.detach().cpu().numpy(), 
                                        global_step, dataformats="HW")
                        writer.add_image("Mel/coarse", 
                                        outputs["spectrogram"]["mel_coarse"][0].T.detach().cpu().numpy(), 
                                        global_step, dataformats="HW")
                        writer.add_image("Mel/fine", 
                                        outputs["spectrogram"]["mel_fine"][0].T.detach().cpu().numpy(), 
                                        global_step, dataformats="HW")

                total_loss += avg_accumulated_loss

                accumulated_loss = 0
                accumulated_loss_dict = {}

                global_step += 1

            if batch_idx % (accumulation_steps * 5) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        num_optimizer_steps = (len(dataloader) + accumulation_steps - 1) // accumulation_steps
        avg_loss = total_loss / max(num_optimizer_steps, 1)

        print(f'Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}')
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'config': config,
                'effective_batch_size': effective_batch_size,
                'accumulation_steps': accumulation_steps,
                'model_type': 'UNet',
                'skip_connections': True
            }, 'best_unet_stuttered_speech_model.pth')
            print(f"New best U-Net model saved with loss: {best_loss:.6f}")

        if epoch % 100 == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'config': config,
                'effective_batch_size': effective_batch_size,
                'accumulation_steps': accumulation_steps,
                'model_type': 'UNet',
                'skip_connections': True
            }, f'unet_checkpoint_epoch_{epoch}.pth')
            print(f"U-Net checkpoint saved at epoch {epoch}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'config': config,
        'effective_batch_size': effective_batch_size,
        'accumulation_steps': accumulation_steps,
        'model_type': 'UNet',
        'skip_connections': True
    }, 'final_unet_stuttered_speech_model.pth')
    print("U-Net training completed!")

def resume_unet_training(checkpoint_path):
    """Resume U-Net training from a saved checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    if checkpoint.get('model_type') != 'UNet':
        print("Warning: Checkpoint doesn't appear to be from U-Net model")

    model = UNetMultitaskStutteredSpeechModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    effective_batch_size = checkpoint.get('effective_batch_size', 8)
    accumulation_steps = checkpoint.get('accumulation_steps', 8)

    base_lr = 1e-4
    lr = base_lr * math.sqrt(effective_batch_size / 8)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        betas=(0.9, 0.98), 
        weight_decay=1e-5, 
        eps=1e-6
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50, 
        T_mult=2, 
        eta_min=1e-6
    )
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']

    print(f"Resumed U-Net training from epoch {start_epoch} with best loss: {best_loss:.6f}")
    print(f"Skip connections enabled: {checkpoint.get('skip_connections', 'Unknown')}")

    return model, optimizer, scheduler, start_epoch, best_loss

def analyze_unet_model(model_path):
    """Analyze the U-Net model architecture and skip connections"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = UNetMultitaskStutteredSpeechModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("=== U-Net Model Analysis ===")
    print(f"Model Type: {checkpoint.get('model_type', 'Unknown')}")
    print(f"Skip Connections: {checkpoint.get('skip_connections', 'Unknown')}")
    print(f"Encoder Layers: {config.num_encoder_layers}")
    print(f"Decoder Layers: {config.num_decoder_layers}")
    print(f"Model Dimension: {config.encoder_dim}")
    print(f"Attention Heads: {config.num_heads}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    print("\n=== Skip Connection Architecture ===")
    print(f"Encoder generates {len(model.encoder.layers)} skip connections")
    print(f"Spectrogram decoder has {len(model.spectrogram_decoder.attention_layers)} layers with skip connections")
    print(f"Transcript decoder has {len(model.transcript_decoder.layers)} layers with skip connections")

    print("\n=== Testing Skip Connections ===")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 100, config.n_mels).to(device)
        encoder_out, skip_connections = model.encoder(dummy_input)
        print(f"Encoder output shape: {encoder_out.shape}")
        print(f"Number of skip connections: {len(skip_connections)}")
        for i, skip in enumerate(skip_connections):
            print(f"Skip connection {i}: {skip.shape}")

    return model, config

if __name__ == "__main__":

    train_unet_model()

