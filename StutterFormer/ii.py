import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import argparse
from torchsummary import summary
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

            if 'effective_batch_size' in checkpoint:
                print(f"  - Trained with effective batch size: {checkpoint['effective_batch_size']}")
            if 'accumulation_steps' in checkpoint:
                print(f"  - Gradient accumulation steps: {checkpoint['accumulation_steps']}")
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

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model loaded with {total_params:,} total parameters ({trainable_params:,} trainable)")

    def _build_vocab(self):
        """Build character vocabulary - must match training exactly"""
        allowed_chars = "abcdefghijklmnopqrstuvwxyz '"

        char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        idx_to_char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}

        for i, char in enumerate(allowed_chars, start=3):
            char_to_idx[char] = i
            idx_to_char[i] = char

        print(f"Vocabulary size: {len(char_to_idx)} (expected: {self.config.vocab_size})")
        return char_to_idx, idx_to_char

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:

            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.to(self.device)

            if sr != self.config.sample_rate:
                resample = torchaudio.transforms.Resample(sr, self.config.sample_rate).to(self.device)
                waveform = resample(waveform)
                print(f"Resampled from {sr} Hz to {self.config.sample_rate} Hz")

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                print("Converted stereo to mono")

            mel_spec = self.mel_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-8)  

            result = mel_spec.squeeze(0).transpose(0, 1)  

            print(f"Extracted mel spectrogram: {result.shape} (time, n_mels)")

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

        if input_mel.shape[0] > self.config.max_audio_length:
            print(f"Truncating audio from {input_mel.shape[0]} to {self.config.max_audio_length} frames")
            input_mel = input_mel[:self.config.max_audio_length]

        print(f"Input mel shape: {input_mel.shape}")

        input_mel = input_mel.unsqueeze(0)  

        print("Running model inference...")
        outputs = self.model(input_mel)
        print(f"Model outputs keys: {outputs.keys()}")

        spectrogram_dict = outputs['spectrogram']  

        transcript_logits = outputs['transcript']

        print(f"Spectrogram dict keys: {spectrogram_dict.keys()}")

        fluent_mel = spectrogram_dict['mel_output']  

        print(f"Generated fluent mel shape: {fluent_mel.shape}")

        transcript_tokens = torch.argmax(transcript_logits, dim=-1)
        transcript = self._decode_transcript(transcript_tokens)
        print(f"Generated transcript: '{transcript}'")

        print("Converting mel to audio...")
        fluent_audio = self._mel_to_audio(fluent_mel)
        print(f"Generated audio shape: {fluent_audio.shape}")

        return {
            'fluent_mel': fluent_mel.squeeze(0).cpu(),  

            'fluent_audio': fluent_audio.cpu(),
            'transcript': transcript,
            'input_mel': input_mel[0].cpu(),
            'mel_coarse': spectrogram_dict['mel_coarse'].squeeze(0).cpu(),
            'mel_fine': spectrogram_dict['mel_fine'].squeeze(0).cpu(),
            'stop_tokens': spectrogram_dict['stop_tokens'].squeeze(0).cpu()
        }

    def save_results(self, results: Dict[str, Any], output_dir: str, filename_prefix: str):
        """Save inference results to files"""
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
        np.save(output_dir / f"{filename_prefix}_mel_coarse.npy", results['mel_coarse'].numpy())
        np.save(output_dir / f"{filename_prefix}_mel_fine.npy", results['mel_fine'].numpy())

        spectrograms_to_plot = [
            (results['input_mel'], "Input Mel Spectrogram", f"{filename_prefix}_input_mel.png"),
            (results['fluent_mel'], "Predicted Fluent Mel Spectrogram", f"{filename_prefix}_fluent_mel.png"),
            (results['mel_coarse'], "Coarse Mel Prediction", f"{filename_prefix}_mel_coarse.png"),
            (results['mel_fine'], "Fine Mel Prediction", f"{filename_prefix}_mel_fine.png")
        ]

        for mel, title, filename in spectrograms_to_plot:
            plt.figure(figsize=(12, 6))
            plt.imshow(mel.numpy().T, origin='lower', aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Mel Frequency Bin")
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        spectrograms = [
            (results['input_mel'], "Input Mel"),
            (results['fluent_mel'], "Predicted Fluent Mel"),
            (results['mel_coarse'], "Coarse Prediction"),
            (results['mel_fine'], "Fine Prediction")
        ]

        for i, (mel, title) in enumerate(spectrograms):
            im = axes[i].imshow(mel.numpy().T, origin='lower', aspect='auto', cmap='viridis')
            axes[i].set_title(title)
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Mel Frequency Bin")
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_prefix}_mel_comprehensive.png", dpi=150, bbox_inches='tight')
        plt.close()

        if 'stop_tokens' in results:
            plt.figure(figsize=(12, 4))
            stop_probs = torch.sigmoid(results['stop_tokens']).numpy()  

            plt.plot(stop_probs)
            plt.title("Stop Token Probabilities")
            plt.xlabel("Time Frame")
            plt.ylabel("Stop Probability")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / f"{filename_prefix}_stop_tokens.png", dpi=150, bbox_inches='tight')
            plt.close()

        summary_path = output_dir / f"{filename_prefix}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Stuttered Speech Conversion Results\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Input Audio: {filename_prefix}\n")
            f.write(f"Generated Transcript: '{results['transcript']}'\n")
            f.write(f"Input Mel Shape: {results['input_mel'].shape}\n")
            f.write(f"Output Mel Shape: {results['fluent_mel'].shape}\n")
            f.write(f"Output Audio Shape: {results['fluent_audio'].shape}\n")
            f.write(f"Sample Rate: {self.config.sample_rate} Hz\n")
            f.write(f"Audio Duration: {results['fluent_audio'].shape[-1] / self.config.sample_rate:.2f} seconds\n")
            f.write(f"Model Config:\n")
            f.write(f"  - Encoder Dim: {self.config.encoder_dim}\n")
            f.write(f"  - Decoder Dim: {self.config.decoder_dim}\n")
            f.write(f"  - Attention Dim: {self.config.attention_dim}\n")
            f.write(f"  - N Mels: {self.config.n_mels}\n")
            f.write(f"  - Max Audio Length: {self.config.max_audio_length}\n")

        print(f"All results saved to: {output_dir}")

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

        print("Initializing inference model...")
        inference = StutteredSpeechInference(args.model_path, args.device)

        print("Running speech conversion...")
        results = inference.convert_speech(args.input_audio)

        input_filename = Path(args.input_audio).stem

        print("Saving results...")
        inference.save_results(results, args.output_dir, input_filename)

        print(f"\n{'='*60}")
        print(f"CONVERSION RESULTS")
        print(f"{'='*60}")
        print(f"Input audio: {args.input_audio}")
        print(f"Generated transcript: '{results['transcript']}'")
        print(f"Output directory: {args.output_dir}")
        print(f"Input mel shape: {results['input_mel'].shape}")
        print(f"Fluent mel shape: {results['fluent_mel'].shape}")
        print(f"Fluent audio shape: {results['fluent_audio'].shape}")
        print(f"Audio duration: {results['fluent_audio'].shape[-1] / inference.config.sample_rate:.2f} seconds")
        print(f"Processing completed successfully!")
        print(f"{'='*60}")

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

    print("Initializing model for batch inference...")
    inference = StutteredSpeechInference(model_path, device)

    input_path = Path(input_dir)
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"**/*{ext}"))  

    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print(f"No audio files found in {input_dir}")
        return

    results_summary = []

    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file}")

            results = inference.convert_speech(str(audio_file))

            filename_prefix = f"{audio_file.stem}"

            inference.save_results(results, output_dir, filename_prefix)

            print(f"Generated transcript: '{results['transcript']}'")

            results_summary.append({
                'file': str(audio_file),
                'transcript': results['transcript'],
                'audio_duration': results['fluent_audio'].shape[-1] / inference.config.sample_rate,
                'input_shape': str(results['input_mel'].shape),
                'output_shape': str(results['fluent_mel'].shape),
                'success': True
            })

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            results_summary.append({
                'file': str(audio_file),
                'error': str(e),
                'success': False
            })
            continue

    summary_path = Path(output_dir) / "batch_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Batch Inference Summary\n")
        f.write("=" * 50 + "\n\n")

        successful = sum(1 for r in results_summary if r['success'])
        f.write(f"Total files: {len(results_summary)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {len(results_summary) - successful}\n\n")

        total_duration = sum(r.get('audio_duration', 0) for r in results_summary if r['success'])
        f.write(f"Total processed audio duration: {total_duration:.2f} seconds\n\n")

        f.write("Individual Results:\n")
        f.write("-" * 30 + "\n")

        for result in results_summary:
            f.write(f"File: {result['file']}\n")
            if result['success']:
                f.write(f"Transcript: '{result['transcript']}'\n")
                f.write(f"Duration: {result['audio_duration']:.2f}s\n")
                f.write(f"Input shape: {result['input_shape']}\n")
                f.write(f"Output shape: {result['output_shape']}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            f.write("-" * 30 + "\n")

    print(f"\nBatch inference completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"Successfully processed: {successful}/{len(results_summary)} files")

if __name__ == "__main__":

    main()