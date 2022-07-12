import numpy as np
import torch
from speechbrain.pretrained import EncoderASR

from app.config import logger

LANG_ID = "fr"
MODEL_PATH = "./pretrained_models/asr-wav2vec2-commonvoice-fr"
TEMP_PATH = "./pretrained_models/EncoderASR_temp"
TARGET_SR = 8_000

# MODEL_PATH = "./pretrained_models/asr-crdnn-commonvoice-fr"
# TARGET_SR = 16_000

# Initialize Speeech Recognition model
# CommonVoice6.1-Fr  WER 9.96 (self-reported)
asr_model = EncoderASR.from_hparams(source=MODEL_PATH, savedir=TEMP_PATH)


def transcribe(speech_arr: np.array, sr: int):
    duration = len(speech_arr) / sr
    signal = torch.tensor(speech_arr)
    signal_norm = asr_model.audio_normalizer(signal, sr)
    batch = signal_norm.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    # response = str(signal.shape)
    # logger.debug(signal_norm)
    logger.debug(f"Rel_length: {rel_length}")
    logger.debug(f"Signal: {batch}")
    predicted_words, predicted_tokens = asr_model.transcribe_batch(
        batch, rel_length
    )
    logger.debug(f"Predicted words: {predicted_words}")
    return predicted_words[0].lower(), duration
