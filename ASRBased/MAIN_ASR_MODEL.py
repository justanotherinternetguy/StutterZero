import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import numpy as np
from IPython.display import Audio
import wave
import pandas as pd
import cv2
import pandas as pd
import matplotlib.pyplot as plt

%load_ext tensorboard
audio_fp = "/home/alien/Git/DATA/LibriStutterData/LibriStutter_16kHz/"
transcript_fp = "/home/alien/Git/DATA/LibriStutterData/LibriStutter Transcripts/"
fp = "/home/alien/Git/XSpeech/data_processing/output16.csv"
df = pd.read_csv(fp)
df
print(df.iloc[0,1])
from scipy.io import wavfile

def get_sampling_rate(filename):
    sampling_rate, _ = wavfile.read(filename)
    return sampling_rate

filename = df.iloc[0,0]
sampling_rate = get_sampling_rate(filename)
print(f'Sampling Rate: {sampling_rate} Hz')
print(df.iloc[:,1][0])
df_list_fps = df.iloc[:,0].tolist()
df_list_text = df.iloc[:,1].tolist()
df_list_text

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")

input_str = "Hello, World!"
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
def load_wav(filename, sr=16000):
    y, sr = librosa.load(filename, sr=sr)
    return y


y = load_wav(df.iloc[0,0])
y.shape

def prepare_dataset(fp, text_fp): #given one
    with open(text_fp, 'r') as file:
        text = file.read()
    input_features = feature_extractor(load_wav(fp), sampling_rate=16000).input_features[0]

    # encode target text to label ids 
    labels = tokenizer(text).input_ids
    print(text)
    return input_features, labels

len(df_list_text)

len(df_list_fps)

batch = []
# each dict 
for i in range(len(df_list_fps)):
    f, l = prepare_dataset(df_list_fps[i], df_list_text[i])
    batch.append({
        "input_features": f,
        "labels": l
    })

print(batch[0])

f, l = prepare_dataset(df_list_fps[3], df_list_text[3])
print(f.shape)
print(len(l))
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.generation_config.language = "english"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
import evaluate

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# from transformers import Seq2SeqTrainingArguments

# training_args = Seq2SeqTrainingArguments(
#     output_dir="./whisper-small-en-sep28",
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=1,
#     learning_rate=1e-5,
#     warmup_steps=500,
#     max_steps=5000,
#     gradient_checkpointing=True,
#     fp16=True,
#     evaluation_strategy="steps",
#     per_device_eval_batch_size=8,
#     predict_with_generate=True,
#     generation_max_length=225,
#     save_steps=1000,
#     eval_steps=1000,
#     logging_steps=25,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=True,
# )


# from transformers import Seq2SeqTrainingArguments

# training_args = Seq2SeqTrainingArguments(
#     output_dir="./whisper-small-sep28",
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=1,
#     learning_rate=1e-5,
#     warmup_steps=500,
#     max_steps=5910,
#     gradient_checkpointing=True,
#     fp16=True,
#     evaluation_strategy="steps",
#     per_device_eval_batch_size=8,
#     predict_with_generate=True,
#     generation_max_length=225,
#     save_steps=1000,
#     eval_steps=1000,
#     logging_steps=25,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=True,
#     num_train_epochs=30
# )


from transformers import Seq2SeqTrainingArguments

# steps_per_epoch = 375
# max_steps = int(steps_per_epoch * 75)
max_steps = 10000

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-sep28",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=max_steps,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

from sklearn.model_selection import train_test_split

train_batch, test_batch = train_test_split(batch, test_size=0.2, random_state=42)

print(len(train_batch))
print(len(test_batch))
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_batch,
    eval_dataset=test_batch,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
!export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trainer.train()

kwargs = {
    "dataset": "SEP-28K",
    "dataset_args": "config: en, split: test",
    "language": "en",
    "model_name": "Whisper-Small Augmented for SEP-28k",
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs)