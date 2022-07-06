import torch
import librosa
import base64
import numpy as np
import logging.config

from speechbrain.pretrained import EncoderASR, EncoderDecoderASR
# from speechbrain.lobes.models.RNNLM import RNNLM
from config import LogConfig

logging.config.dictConfig(LogConfig().dict())
logger = logging.getLogger("voice-transcript")

LANG_ID = "fr"
MODEL_PATH = "./pretrained_models/asr-wav2vec2-commonvoice-fr"
#MODEL_PATH = "./pretrained_models/asr-crdnn-commonvoice-fr"
TARGET_SR = 8_000
#TARGET_SR = 16_000

# Initialize Speeech Recognition model
# CommonVoice6.1-Fr  WER 9.96 (self-reported)
#asr_model = EncoderASR.from_hparams(source=MODEL_PATH)
asr_model = EncoderDecoderASR.from_hparams(source=MODEL_PATH)


def transcribe(speech_arr: np.array, sr):
    duration = len(speech_arr) / sr
    signal = torch.tensor(speech_arr)
    signal_norm = asr_model.audio_normalizer(signal, sr)
    batch = signal_norm.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    # response = str(signal.shape)
    # logger.debug(signal_norm)
    predicted_words, predicted_tokens = asr_model.transcribe_batch(batch,
                                                                   rel_length
                                                                   )
    # logger.debug(f"Rel_length: {rel_length}")
    logger.debug(f"Predicted words: {predicted_words}")
    return predicted_words[0].lower(), duration
