from transformers import pipeline
import numpy as np
import jiwer
import pandas as pd
import sys
# pipe = pipeline(task="automatic-speech-recognition", model="openai/whisper-medium")
pipe = pipeline(task="automatic-speech-recognition", model="justanotherinternetguy/whisper-small-sep28")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

def read_ground_truth(ground_truth_file):
    with open(ground_truth_file, "r") as file:
        ground_truth = file.read().strip()
    return ground_truth

def calculate_wer(transcribed_text, ground_truth_text):
    return jiwer.wer(ground_truth_text, transcribed_text)

def calculate_cer(transcribed_text, ground_truth_text):
    return jiwer.cer(ground_truth_text, transcribed_text)

def calculate_mer(transcribed_text, ground_truth_text):
    return jiwer.mer(ground_truth_text, transcribed_text)

def calculate_wil(transcribed_text, ground_truth_text):
    return jiwer.wil(ground_truth_text, transcribed_text)
    

def calculate_wip(transcribed_text, ground_truth_text):
    return jiwer.wip(ground_truth_text, transcribed_text)
    

csv_path = '/home/alien/Git/XSpeech/data_processing/Libristutter_16hkz_fps.csv'

df = pd.read_csv(csv_path)

wers = []
mers = []
wils = []
wips = []
cers = []

with open('transcription_results.txt', 'w') as output_file:
    sys.stdout = output_file

    for index, row in df.iterrows():
        audio_path = row['stuttered_fp']
        ground_truth_path = row['transcript_fp']

        transcribed_text = transcribe(audio_path)

        ground_truth_text = read_ground_truth(ground_truth_path)

        wer = calculate_wer(transcribed_text, ground_truth_text)
        wers.append(wer)

        mer = calculate_mer(transcribed_text, ground_truth_text)
        mers.append(mer)

        wil = calculate_wil(transcribed_text, ground_truth_text)
        wils.append(wil)

        wip = calculate_wip(transcribed_text, ground_truth_text)
        wips.append(wip)

        cer = calculate_cer(transcribed_text, ground_truth_text)
        cers.append(cer)

        print(f"Audio File: {audio_path}")
        print(f"Transcribed Text: {transcribed_text}")
        print(f"Ground Truth Text: {ground_truth_text}")
        print(f"WER: {wer}")
        print(f"MER: {mer}")
        print(f"WIL: {wil}")
        print(f"WIP: {wip}")
        print(f"CER: {cer}")
        print("="*50)

    mean_wer = np.mean(np.array(wers))
    mean_mer = np.mean(np.array(mers))
    mean_wil = np.mean(np.array(wils))
    mean_wip = np.mean(np.array(wips))
    mean_cer = np.mean(np.array(cers))

    print(f"Mean WER: {mean_wer}")
    print(f"Mean MER: {mean_mer}")
    print(f"Mean WIL: {mean_wil}")
    print(f"Mean WIP: {mean_wip}")
    print(f"Mean CER: {mean_cer}")
    
sys.stdout = sys.__stdout__
