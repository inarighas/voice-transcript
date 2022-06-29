import torch
import librosa
import base64
import numpy as np
import logging.config

from spellchecker import SpellChecker
from speechbrain.pretrained import EncoderASR
from config import LogConfig

logging.config.dictConfig(LogConfig().dict())
logger = logging.getLogger("voice-transcript")


# Initialize Speeech Recognition model
# CommonVoice6.1-Fr  WER 9.96 (self-reported)
asr_model = EncoderASR.from_hparams(
    source="./pretrained_models/asr-wav2vec2-commonvoice-fr"
)
TARGET_SR = 8000
spell = SpellChecker(language='fr')


def speech_to_array_fn(audio, origin="buffer"):
    if origin == "buffer":
        arr = base64.b64decode(audio)
        speech_array = np.frombuffer(arr, dtype=np.float32)
    elif origin == "file":
        tmp, orig_sr = librosa.load('samples/' + audio + '.wav')
        speech_array = librosa.resample(tmp, orig_sr=orig_sr,
                                        target_sr=TARGET_SR)

    return speech_array
# CommonVoice6.1-fr : WER 17.62  CER 6.040 (self-reported)
# asr_model2 = EncoderASR.from_hparams(
#     source="./pretrained_models/wav2vec2-large-fr-voxpopuli-french"
# )

# CommonVoice-fr : WER 17.62  CER 6.040 (self-reported)
# asr_model3 = EncoderASR.from_hparams(
#     source="./pretrained_models/wav2vec2-large-fr-voxpopuli-french"
# )


def transcribe(input, sr, origin="file"):
    speech_arr = speech_to_array_fn(input, origin=origin)
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
    # tmp = spell.unknown(str(predicted_words[0]).split())
    # response = [spell.correction(w) for w in tmp]
    # logger.debug(f"Rel_length: {rel_length}")
    logger.debug(f"Predicted words: {predicted_words}")
    return predicted_words[0], duration
