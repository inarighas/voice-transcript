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
spell = SpellChecker(language='fr')


# asr_model = SpeechRecognitionModel(MODEL_ID)
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# # Preprocessing the datasets.
# # We need to read the audio files as arrays


def speech_to_array_fn(audio, origin="buffer"):
    if origin == "buffer":
        arr = base64.b64decode(audio)
        speech_array = np.frombuffer(arr, dtype=np.float32)
    elif origin == "file":
        tmp, orig_sr = librosa.load('samples/' + audio + '.wav')
        speech_array = librosa.resample(tmp, orig_sr=orig_sr,
                                        target_sr=TARGET_SR)

    return speech_array


def transcribe(input: str | bytes, sr: int, origin="file"):
    speech_arr = speech_to_array_fn(input, origin=origin)
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
    #logger.debug(f"pred Ids: {predicted_ids}")
    predicted_sentences = processor.batch_decode(predicted_ids)
    logger.debug(f"prd: {predicted_sentences}")
    tmp = spell.unknown(predicted_sentences[0].split())
    logger.debug(f"spell tmp: {tmp}")
    response = [spell.correction(w) for w in predicted_sentences[0].split()]
    logger.debug(f"spell res: {response}")
    return ' '.join(response), duration
