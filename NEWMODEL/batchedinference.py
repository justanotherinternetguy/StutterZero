import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
import os
from typing import Optional, Dict, Any
import soundfile as sf
from pathlib import Path

from other import (
    Config, 
    MultitaskStutteredSpeechModel,
    PositionalEncoding,
    MultiHeadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    SpeechEncoder,
    SpectrogramDecoder,
    TranscriptDecoder
)

def safe_torch_load(model_path: str, device: torch.device):
    """
    Safely load a PyTorch checkpoint with multiple fallback methods
    for compatibility across PyTorch versions
    """

    try:
        import torch.serialization

        try:
            from other import Config as ConfigClass
        except ImportError:

            class ConfigClass:
                pass

        with torch.serialization.safe_globals([ConfigClass]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            print("✓ Loaded with safe globals (PyTorch 2.6+ secure method)")
            return checkpoint
    except Exception as e:
        print(f"Safe loading failed: {str(e)[:100]}...")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("✓ Loaded with weights_only=False (compatible method)")
        return checkpoint
    except Exception as e:
        print(f"weights_only=False failed: {str(e)[:100]}...")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("✓ Loaded with default torch.load (legacy method)")
        return checkpoint
    except Exception as e:
        print(f"Default torch.load failed: {str(e)[:100]}...")

    raise RuntimeError(
        f"Could not load checkpoint from {model_path}. "
        f"This might be due to PyTorch version compatibility issues. "
        f"Try using PyTorch 2.5 or earlier, or ensure the checkpoint was saved properly."
    )

class StutteredSpeechInference:
    """Simplified inference class for stuttered speech conversion"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize inference model

        Args:
            model_path: Path to the saved model checkpoint
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")

        print(f"Loading model from: {model_path}")
        checkpoint = safe_torch_load(model_path, self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f"📊 Checkpoint info:")
            print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  - Best training loss: {checkpoint.get('best_loss', 'unknown'):.6f}")

            if 'config' in checkpoint:
                self.config = checkpoint['config']
                print("  - Using config from checkpoint")
            else:
                self.config = Config()
                print("  - Using default config (checkpoint doesn't contain config)")
        else:

            self.config = Config()
            print("📊 Using default config (simple state dict format)")

        self.model = MultitaskStutteredSpeechModel(self.config).to(self.device)

        try:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:

                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Model weights loaded from checkpoint format successfully!")
            elif isinstance(checkpoint, dict) and any(key.startswith(('encoder.', 'spectrogram_decoder.', 'transcript_decoder.')) for key in checkpoint.keys()):

                self.model.load_state_dict(checkpoint)
                print("Model weights loaded from state dict format successfully!")
            else:

                self.model.load_state_dict(checkpoint)
                print("Model weights loaded (fallback method) successfully!")

        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Available keys in checkpoint:")
            if isinstance(checkpoint, dict):
                for key in list(checkpoint.keys())[:10]:  

                    print(f"  - {key}")
                if len(checkpoint.keys()) > 10:
                    print(f"  ... and {len(checkpoint.keys()) - 10} more keys")
            raise RuntimeError(f"Could not load model weights: {e}")

        self.model.eval()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        ).to(self.device)

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            power=1.0  

        ).to(self.device)

        self.inverse_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=self.config.n_fft // 2 + 1,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model loaded with {total_params:,} total parameters ({trainable_params:,} trainable)")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:

            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.to(self.device)

            if sr != self.config.sample_rate:
                resample = torchaudio.transforms.Resample(sr, self.config.sample_rate).to(self.device)
                waveform = resample(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            mel_spec = self.mel_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-8)  

            result = mel_spec.squeeze(0).transpose(0, 1)  

            return result

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            raise

    def _mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram back to audio using Griffin-Lim"""

        if mel_spec.dim() == 3:  

            mel_spec = mel_spec.squeeze(0)  

        mel_spec = mel_spec.transpose(0, 1).unsqueeze(0)  

        mel_magnitude = torch.exp(mel_spec)

        try:

            linear_magnitude = self.inverse_mel_scale(mel_magnitude)
        except Exception as e:

            linear_magnitude = F.interpolate(
                mel_magnitude, 
                size=(self.config.n_fft // 2 + 1, mel_magnitude.size(-1)), 
                mode='bilinear', 
                align_corners=False
            )

        linear_magnitude = linear_magnitude.squeeze(0)  

        audio = self.griffin_lim(linear_magnitude)

        return audio

    @torch.no_grad()
    def convert_speech(self, input_audio_path: str) -> torch.Tensor:
        """
        Convert stuttered speech to fluent speech

        Args:
            input_audio_path: Path to input stuttered audio file

        Returns:
            fluent_audio: Generated fluent speech audio waveform
        """

        input_mel = self._load_audio(input_audio_path)

        if input_mel.shape[0] > self.config.max_audio_length:
            input_mel = input_mel[:self.config.max_audio_length]

        input_mel = input_mel.unsqueeze(0)  

        outputs = self.model(input_mel)

        spectrogram_dict = outputs['spectrogram']  

        fluent_mel = spectrogram_dict['mel_output']  

        fluent_audio = self._mel_to_audio(fluent_mel)

        return fluent_audio.cpu()

def main():
    """Main function to process CSV input and generate fluent audio"""

    model_path = "best_stuttered_speech_model.pth"  

    csv_path = "/home/alien/Git/StutterZero/EEModel/label,csv/train3.csv"
    output_dir = "./out"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        if 'stuttered_speech' not in df.columns:
            print(f"Error: Column 'stuttered_speech' not found in CSV. Available columns: {df.columns.tolist()}")
            return

        print(f"Found {len(df)} files to process in CSV")

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    try:
        print("Initializing inference model...")
        inference = StutteredSpeechInference(model_path)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    successful = 0
    failed = 0

    for idx, row in df.iterrows():
        input_audio_path = row['stuttered_speech']

        if not os.path.exists(input_audio_path):
            print(f"[{idx+1}/{len(df)}] Error: Input file not found: {input_audio_path}")
            failed += 1
            continue

        try:
            print(f"[{idx+1}/{len(df)}] Processing: {input_audio_path}")

            fluent_audio = inference.convert_speech(input_audio_path)

            input_filename = Path(input_audio_path).name

            output_filename = Path(input_filename).stem + Path(input_filename).suffix
            output_path = Path(output_dir) / output_filename

            sf.write(
                output_path, 
                fluent_audio.numpy(), 
                inference.config.sample_rate
            )

            print(f"    Saved: {output_path}")
            successful += 1

        except Exception as e:
            print(f"[{idx+1}/{len(df)}] Error processing {input_audio_path}: {e}")
            failed += 1
            continue

    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files: {len(df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    summary_path = Path(output_dir) / "processing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Stuttered Speech Conversion Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Input CSV: {csv_path}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Total files: {len(df)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Sample rate: {inference.config.sample_rate} Hz\n")

    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()