import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from unet import (
    UNetMultitaskStutteredSpeechModel, 
    Config,
    PositionalEncoding,
    MultiHeadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    UNetSpeechEncoder,
    UNetSpectrogramDecoder,
    TranscriptDecoder,
    ResidualBlock
)

class InferenceConfig:
    """Configuration for inference"""
    def __init__(self):

        self.sample_rate = 16000
        self.n_mels = 128
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024

        self.encoder_dim = 256
        self.decoder_dim = 256
        self.attention_dim = 128
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.num_heads = 4
        self.dropout = 0.3

        self.vocab_size = 31
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2

        self.max_audio_length = 800
        self.max_text_length = 200

class UNetStutteredSpeechInference:
    """Inference class for U-Net stuttered speech conversion model"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the inference engine

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")

        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = self.checkpoint['config']

        if self.checkpoint.get('model_type') != 'UNet':
            print("Warning: Model type is not explicitly marked as U-Net")

        print(f"Loaded model from epoch {self.checkpoint['epoch']} with loss {self.checkpoint.get('best_loss', 'Unknown')}")
        print(f"Skip connections enabled: {self.checkpoint.get('skip_connections', 'Unknown')}")

        self.model = UNetMultitaskStutteredSpeechModel(self.config).to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        ).to(self.device)

        self.inverse_mel_transform = torchaudio.transforms.InverseMelScale(
            n_stft=self.config.n_fft // 2 + 1,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate
        ).to(self.device)

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_iter=60
        ).to(self.device)

        self.char_to_idx, self.idx_to_char = self._build_vocab_mappings()

        print("Model loaded successfully!")
        self._print_model_info()

    def _build_vocab_mappings(self):
        """Build character-to-index and index-to-character mappings"""
        allowed_chars = "abcdefghijklmnopqrstuvwxyz '"
        char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        idx_to_char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}

        for i, char in enumerate(allowed_chars, start=3):
            char_to_idx[char] = i
            idx_to_char[i] = char

        return char_to_idx, idx_to_char

    def _print_model_info(self):
        """Print model architecture information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel Architecture:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Encoder layers: {self.config.num_encoder_layers}")
        print(f"  - Decoder layers: {self.config.num_decoder_layers}")
        print(f"  - Model dimension: {self.config.encoder_dim}")
        print(f"  - Attention heads: {self.config.num_heads}")
        print(f"  - Mel spectrogram bins: {self.config.n_mels}")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed mel spectrogram tensor
        """
        try:

            waveform, sr = torchaudio.load(audio_path)

            if sr != self.config.sample_rate:
                resample = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resample(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.to(self.device)

            mel_spec = self.mel_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-8)  

            mel_spec = mel_spec.squeeze(0).transpose(0, 1)

            return mel_spec

        except Exception as e:
            raise Exception(f"Error loading audio {audio_path}: {e}")

    def _mel_to_audio(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram back to audio waveform
        """

        mel_spec = torch.exp(mel_spectrogram) - 1e-8

        mel_spec = mel_spec.transpose(0, 1).unsqueeze(0)

        linear_spec = self.inverse_mel_transform(mel_spec)

        waveform = self.griffin_lim(linear_spec)

        return waveform.squeeze(0)

    def _decode_transcript(self, token_indices: torch.Tensor) -> str:
        """
        Decode token indices back to text

        Args:
            token_indices: Tensor of token indices

        Returns:
            Decoded text string
        """
        text = ""
        for idx in token_indices:
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            if idx_val == self.config.eos_token:
                break
            elif idx_val in self.idx_to_char and idx_val not in [self.config.pad_token, self.config.sos_token]:
                text += self.idx_to_char[idx_val]
        return text.strip()

    def infer_single(self, audio_path: str, return_transcript: bool = True) -> Dict[str, Any]:
        """
        Perform inference on a single audio file

        Args:
            audio_path: Path to stuttered speech audio file
            return_transcript: Whether to also generate transcript

        Returns:
            Dictionary containing inference results
        """
        with torch.no_grad():

            input_mel = self._load_audio(audio_path)

            input_mel_batch = input_mel.unsqueeze(0)  

            print(f"Input audio shape: {input_mel_batch.shape}")

            outputs = self.model(input_mel_batch)

            result = {}

            output_mel = outputs['spectrogram']['mel_output'].squeeze(0)  

            result['fluent_mel'] = output_mel.cpu()

            try:
                fluent_audio = self._mel_to_audio(output_mel)
                result['fluent_audio'] = fluent_audio.cpu()
            except Exception as e:
                print(f"Warning: Could not convert mel to audio: {e}")
                result['fluent_audio'] = None

            if return_transcript:
                transcript_tokens = outputs['transcript'].argmax(dim=-1).squeeze(0)
                transcript_text = self._decode_transcript(transcript_tokens)
                result['transcript'] = transcript_text
                result['transcript_tokens'] = transcript_tokens.cpu()

            if 'stop_tokens' in outputs['spectrogram']:
                result['stop_tokens'] = outputs['spectrogram']['stop_tokens'].squeeze(0).cpu()

            if 'mel_coarse' in outputs['spectrogram']:
                result['mel_coarse'] = outputs['spectrogram']['mel_coarse'].squeeze(0).cpu()
                result['mel_fine'] = outputs['spectrogram']['mel_fine'].squeeze(0).cpu()

            print(f"Inference completed. Output shape: {output_mel.shape}")
            if return_transcript:
                print(f"Generated transcript: '{result['transcript']}'")

            return result

    def infer_batch(self, audio_paths: List[str], return_transcripts: bool = True) -> List[Dict[str, Any]]:
        """
        Perform inference on multiple audio files

        Args:
            audio_paths: List of paths to stuttered speech audio files
            return_transcripts: Whether to also generate transcripts

        Returns:
            List of dictionaries containing inference results
        """
        results = []

        print(f"Processing {len(audio_paths)} audio files...")

        for i, audio_path in enumerate(audio_paths):
            print(f"\nProcessing file {i+1}/{len(audio_paths)}: {audio_path}")
            try:
                result = self.infer_single(audio_path, return_transcripts)
                result['input_path'] = audio_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'input_path': audio_path,
                    'error': str(e)
                })

        return results

    def save_results(self, results: Dict[str, Any], output_dir: str, base_name: str = "output"):
        os.makedirs(output_dir, exist_ok=True)

        if results.get('fluent_audio') is not None:
            audio_path = os.path.join(output_dir, f"{base_name}_fluent.wav")
            sf.write(audio_path, results['fluent_audio'].numpy(), self.config.sample_rate)
            print(f"Saved fluent audio: {audio_path}")

        if 'fluent_mel' in results and results['fluent_mel'] is not None:
            try:

                waveform_from_mel = self._mel_to_audio(results['fluent_mel'].to(self.device))
                spec_audio_path = os.path.join(output_dir, f"{base_name}_from_spectrogram.wav")
                sf.write(spec_audio_path, waveform_from_mel.cpu().numpy(), self.config.sample_rate)
                print(f"Saved audio reconstructed from spectrogram: {spec_audio_path}")
            except Exception as e:
                print(f"Warning: Could not save spectrogram audio: {e}")

        if 'transcript' in results:
            transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
            with open(transcript_path, 'w') as f:
                f.write(results['transcript'])
            print(f"Saved transcript: {transcript_path}")

        if 'fluent_mel' in results:
            plt.figure(figsize=(12, 6))
            plt.imshow(results['fluent_mel'].T.numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.title('Generated Fluent Speech Mel Spectrogram')
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Frequency Bins')
            plt.colorbar(label='Log Magnitude')
            mel_plot_path = os.path.join(output_dir, f"{base_name}_mel_spectrogram.png")
            plt.savefig(mel_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved mel spectrogram: {mel_plot_path}")

        info_path = os.path.join(output_dir, f"{base_name}_info.txt")
        with open(info_path, 'w') as f:
            f.write("=== Inference Results ===\n")
            f.write(f"Model: {self.checkpoint.get('model_type', 'Unknown')}\n")
            f.write(f"Epoch: {self.checkpoint['epoch']}\n")
            f.write(f"Best Loss: {self.checkpoint.get('best_loss', 'Unknown')}\n")
            if 'transcript' in results:
                f.write(f"Transcript: {results['transcript']}\n")
            if 'fluent_mel' in results:
                f.write(f"Output mel shape: {results['fluent_mel'].shape}\n")
            if 'fluent_audio' in results and results['fluent_audio'] is not None:
                f.write(f"Output audio duration: {len(results['fluent_audio']) / self.config.sample_rate:.2f} seconds\n")
        print(f"Saved info: {info_path}")

def main():
    parser = argparse.ArgumentParser(description="U-Net Stuttered Speech Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint (.pth file)")
    parser.add_argument("--audio_path", type=str,
                       help="Path to input stuttered speech audio file")
    parser.add_argument("--audio_list", type=str,
                       help="Path to text file containing list of audio files")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs",
                       help="Directory to save outputs")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default=None,
                       help="Device to run inference on")
    parser.add_argument("--no_transcript", action="store_true",
                       help="Skip transcript generation")

    args = parser.parse_args()

    if not args.audio_path and not args.audio_list:
        parser.error("Either --audio_path or --audio_list must be specified")

    if not os.path.exists(args.model_path):
        parser.error(f"Model file not found: {args.model_path}")

    print("Initializing U-Net inference engine...")
    inference_engine = UNetStutteredSpeechInference(args.model_path, args.device)

    audio_files = []
    if args.audio_path:
        if not os.path.exists(args.audio_path):
            print(f"Error: Audio file not found: {args.audio_path}")
            return
        audio_files = [args.audio_path]

    if args.audio_list:
        if not os.path.exists(args.audio_list):
            print(f"Error: Audio list file not found: {args.audio_list}")
            return
        with open(args.audio_list, 'r') as f:
            audio_files.extend([line.strip() for line in f if line.strip()])

    generate_transcripts = not args.no_transcript

    if len(audio_files) == 1:
        print(f"\nProcessing single file: {audio_files[0]}")
        results = inference_engine.infer_single(audio_files[0], generate_transcripts)

        base_name = Path(audio_files[0]).stem
        inference_engine.save_results(results, args.output_dir, base_name)

    else:
        print(f"\nProcessing {len(audio_files)} files in batch mode...")
        results_list = inference_engine.infer_batch(audio_files, generate_transcripts)

        for i, results in enumerate(results_list):
            if 'error' in results:
                print(f"Skipping failed file: {results['input_path']}")
                continue

            base_name = Path(results['input_path']).stem
            inference_engine.save_results(results, args.output_dir, f"{base_name}_{i:03d}")

    print(f"\nInference completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

"""

python inference.py --model_path best_unet_stuttered_speech_model.pth --audio_path input_stuttered.wav

python inference.py --model_path best_unet_stuttered_speech_model.pth --audio_list audio_files.txt

python inference.py --model_path best_unet_stuttered_speech_model.pth --audio_path input.wav --no_transcript

python inference.py --model_path best_unet_stuttered_speech_model.pth --audio_path input.wav --output_dir ./results --device cuda
"""