import logging.config
import torch
import base64
import librosa
import numpy as np

from spellchecker import SpellChecker
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# from huggingsound import SpeechRecognitionModel
from config import LogConfig

logging.config.dictConfig(LogConfig().dict())
logger = logging.getLogger("voice-transcript")

LANG_ID = "fr"
MODEL_ID = "jonatasgrosman/wav2vec2-large-fr-voxpopuli-french"
TARGET_SR = 16_000


# asr_model = SpeechRecognitionModel(MODEL_ID)
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# # Preprocessing the datasets.
# # We need to read the audio files as arrays

def transcribe(speech_arr: np.array, sr: int):
    duration = len(speech_arr) / sr
    inputs = processor(speech_arr,
                       sampling_rate=TARGET_SR,
                       return_tensors="pt",
                       padding=True
                       )

    with torch.no_grad():
        logits = model(inputs.input_values,
                       attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    return predicted_sentences[0].lower(), duration
