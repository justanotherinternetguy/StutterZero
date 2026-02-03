import pandas as pd
import numpy as np
import melo.api
from melo.api import TTS

device = 'cuda'



text = "what was her?"
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'asr-HeStutters_1_32.wav'
model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=1)
import wave

with wave.open(output_path, 'rb') as wav_file:
    sample_rate = wav_file.getframerate()

print(f"Sample rate: {sample_rate} Hz")


import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

audio_path = "asr-HeStutters_1_32.wav"
y, sr = librosa.load(audio_path, sr=None)

y_resampled = librosa.resample(y, sr, 16000)

y_stretched = librosa.effects.time_stretch(y_resampled, rate=1.05) 
y_pitch_shifted = librosa.effects.pitch_shift(y_stretched, 16000, n_steps=1)

sf.write(f'e2e_{output_path}', y_pitch_shifted, 16000)

csv = "/home/alien/Git/XSpeech/data_processing/transcription_small_tt_formatted.csv"

import csv
import os
import sys
import logging
from melo.api import TTS

logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StreamToLogger(object):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), level=logging.INFO)

device = 'cuda'
model = TTS(language='EN', device=device)

csv_filepath = '/home/alien/Git/XSpeech/data_processing/transcription_small_tt_formatted.csv'

output_directory = '/home/alien/Git/DATA/MeloTTSAudioSmall'

os.makedirs(output_directory, exist_ok=True)

try:
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            audio_filepath = row['filepath']
            text_result = row['result']
            # with open(row['result'], 'r') as file:
                # text_result = file.read()
            
            base_filename = os.path.splitext(os.path.basename(audio_filepath))[0]
            
            output_filepath = os.path.join(output_directory, f'{base_filename}.wav')
            
            speaker_ids = model.hps.data.spk2id
            model.tts_to_file(text_result, speaker_ids['EN-US'], output_filepath)

except Exception as e:
    print(f'An error occurred: {e}')

import csv
import os
import sys
import logging
from melo.api import TTS
import librosa
import soundfile as sf

logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StreamToLogger(object):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), level=logging.INFO)

device = 'cuda'
model = TTS(language='EN', device=device)

csv_filepath = '/home/alien/Git/XSpeech/data_processing/transcription_small_tt_formatted.csv'

output_directory = '/home/alien/Git/DATA/MeloTTSAudioSmall16khz'

os.makedirs(output_directory, exist_ok=True)

def resample_audio(input_path, output_path, target_sr=16000):
    y, sr = librosa.load(input_path, sr=None)
    
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    sf.write(output_path, y_resampled, target_sr, subtype='PCM_16')

try:
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            audio_filepath = row['filepath']
            text_result = row['result']
            
            base_filename = os.path.splitext(os.path.basename(audio_filepath))[0]
            
            temp_output_filepath = os.path.join(output_directory, f'{base_filename}_temp.wav')
            
            final_output_filepath = os.path.join(output_directory, f'{base_filename}.wav')
            
            speaker_ids = model.hps.data.spk2id
            model.tts_to_file(text_result, speaker_ids['EN-US'], temp_output_filepath)
            
            resample_audio(temp_output_filepath, final_output_filepath)
            
            os.remove(temp_output_filepath)
            
            print(f"Processed and resampled: {final_output_filepath}")

except Exception as e:
    print(f'An error occurred: {e}')
