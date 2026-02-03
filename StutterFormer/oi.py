import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import argparse
import os
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
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

class StutteredSpeechInference:
    """Inference class for stuttered speech conversion"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize inference model

        Args:
            model_path: Path to the saved model checkpoint
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")

        self.config = Config()

        self.model = MultitaskStutteredSpeechModel(self.config).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
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

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate,
            n_stft=self.config.n_fft // 2 + 1
        ).to(self.device)

        self.inverse_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=self.config.n_fft // 2 + 1,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate
        ).to(self.device)

        self.char_to_idx, self.idx_to_char = self._build_vocab()

    def _build_vocab(self):
        """Build character vocabulary - simplified version for inference"""

        allowed_chars = "abcdefghijklmnopqrstuvwxyz '"

        char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        idx_to_char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}

        for i, char in enumerate(allowed_chars, start=3):
            char_to_idx[char] = i
            idx_to_char[i] = char

        return char_to_idx, idx_to_char

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

            return mel_spec.squeeze(0).transpose(0, 1)  

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            raise

    def _mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram back to audio using Griffin-Lim"""

        mel_spec = mel_spec.transpose(0, 1).unsqueeze(0)  

        mel_magnitude = torch.exp(mel_spec)

        try:

            linear_magnitude = self.inverse_mel_scale(mel_magnitude)
        except Exception as e:
            print(f"Warning: InverseMelScale failed ({e}), using alternative method")

            linear_magnitude = F.interpolate(
                mel_magnitude, 
                size=(self.config.n_fft // 2 + 1, mel_magnitude.size(-1)), 
                mode='bilinear', 
                align_corners=False
            )

        linear_magnitude = linear_magnitude.squeeze(0)  

        audio = self.griffin_lim(linear_magnitude)

        return audio

    def _decode_transcript(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text"""
        tokens = token_ids.cpu().numpy()

        if tokens.ndim > 1:
            tokens = tokens[0]

        text = ""
        for token_id in tokens:
            if token_id == self.config.eos_token:
                break
            if token_id in self.idx_to_char and token_id not in [0, 1, 2]:  

                text += self.idx_to_char[token_id]

        return text.strip()

    @torch.no_grad()
    def convert_speech(self, input_audio_path: str) -> Dict[str, Any]:
        """
        Convert stuttered speech to fluent speech

        Args:
            input_audio_path: Path to input stuttered audio file

        Returns:
            Dictionary containing:
                - 'fluent_mel': Generated fluent speech mel spectrogram
                - 'fluent_audio': Generated fluent speech audio waveform
                - 'transcript': Generated transcript text
        """
        print(f"Processing: {input_audio_path}")

        input_mel = self._load_audio(input_audio_path)

        input_mel = input_mel.unsqueeze(0)  

        outputs = self.model(input_mel)

        fluent_mel = outputs['spectrogram']  

        transcript_logits = outputs['transcript']

        transcript_tokens = torch.argmax(transcript_logits, dim=-1)
        transcript = self._decode_transcript(transcript_tokens)

        fluent_audio = self._mel_to_audio(fluent_mel)

        return {
            'fluent_mel': fluent_mel.cpu(),
            'fluent_audio': fluent_audio.cpu(),
            'transcript': transcript,
            'input_mel': input_mel[0].cpu()
        }

    import matplotlib.pyplot as plt

    def save_results(self, results: Dict[str, Any], output_dir: str, filename_prefix: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fluent_audio_path = output_dir / f"{filename_prefix}_fluent.wav"
        sf.write(
            fluent_audio_path, 
            results['fluent_audio'].numpy(), 
            self.config.sample_rate
        )
        print(f"Saved fluent audio: {fluent_audio_path}")

        transcript_path = output_dir / f"{filename_prefix}_transcript.txt"
        with open(transcript_path, 'w') as f:
            f.write(results['transcript'])
        print(f"Saved transcript: {transcript_path}")

        np.save(output_dir / f"{filename_prefix}_input_mel.npy", results['input_mel'].numpy())
        np.save(output_dir / f"{filename_prefix}_fluent_mel.npy", results['fluent_mel'].numpy())

        for mel, title, name in [
            (results['input_mel'], "Input Mel Spectrogram", f"{filename_prefix}_input_mel.png"),
            (results['fluent_mel'], "Predicted Fluent Mel Spectrogram", f"{filename_prefix}_fluent_mel.png")
        ]:
            plt.figure(figsize=(10, 4))
            plt.imshow(mel.numpy().T, origin='lower', aspect='auto')
            plt.colorbar()
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_dir / name)
            plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        axes[0].imshow(results['input_mel'].numpy().T, origin='lower', aspect='auto')
        axes[0].set_title("Input Mel")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Mel Frequency Bin")

        axes[1].imshow(results['fluent_mel'].numpy().T, origin='lower', aspect='auto')
        axes[1].set_title("Predicted Mel")
        axes[1].set_xlabel("Time")

        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_prefix}_mel_comparison.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Stuttered Speech Conversion Inference")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.pth file)")
    parser.add_argument("--input_audio", required=True, help="Path to input stuttered audio file")
    parser.add_argument("--output_dir", default="./inference_output", help="Output directory")
    parser.add_argument("--device", default=None, help="Device to use (cpu/cuda)")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return

    if not os.path.exists(args.input_audio):
        print(f"Error: Input audio file not found: {args.input_audio}")
        return

    try:

        inference = StutteredSpeechInference(args.model_path, args.device)

        results = inference.convert_speech(args.input_audio)

        input_filename = Path(args.input_audio).stem

        inference.save_results(results, args.output_dir, input_filename)

        print(f"\n--- Results ---")
        print(f"Input audio: {args.input_audio}")
        print(f"Generated transcript: '{results['transcript']}'")
        print(f"Output directory: {args.output_dir}")
        print(f"Fluent audio shape: {results['fluent_audio'].shape}")
        print(f"Processing completed successfully!")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

def batch_inference(model_path: str, input_dir: str, output_dir: str, device: str = None):
    """
    Run inference on multiple audio files in a directory

    Args:
        model_path: Path to trained model
        input_dir: Directory containing input audio files
        output_dir: Directory to save results
        device: Device to use for inference
    """

    inference = StutteredSpeechInference(model_path, device)

    input_path = Path(input_dir)
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"**/*{ext}"))  

    print(f"Found {len(audio_files)} audio files")

    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file}")

            results = inference.convert_speech(str(audio_file))

            filename_prefix = f"{audio_file.stem}"

            inference.save_results(results, output_dir, filename_prefix)

            print(f"Generated transcript: '{results['transcript']}'")

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

    print(f"\nBatch inference completed! Results saved to: {output_dir}")

if __name__ == "__main__":

    main()