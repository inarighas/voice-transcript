import base64
import logging.config

import librosa
import numpy as np
import torch
from config import LogConfig
from speechbrain.pretrained.interfaces import foreign_class

logging.config.dictConfig(LogConfig().dict())
logger = logging.getLogger("voice-transcript")


clf = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
)

# Initialize Speeech Recognition model
# CommonVoice6.1-Fr  WER 9.96 (self-reported)


def speech_to_array_fn(audio, origin="buffer"):
    if origin == "buffer":
        arr = base64.b64decode(audio)
        speech_array = np.frombuffer(arr, dtype=np.float32)
    elif origin == "file":
        tmp, orig_sr = librosa.load("samples/" + audio + ".wav")
        logger.debug(
            f"File Loaded. Original SR: {orig_sr}, NSamples {tmp.shape}"
        )
        speech_array = librosa.resample(
            tmp,
            orig_sr=orig_sr,
            target_sr=TARGET_SR,
            res_type="kaiser_fast",
        )
    return speech_array


def analyse_sentiment(audio, sr, origin="file"):
    # speech_arr = speech_to_array_fn(audio, origin=origin)
    # duration = len(speech_arr) / sr
    # signal = torch.tensor(speech_arr)
    # batch = signal.unsqueeze(0)
    # rel_length = torch.tensor([1.0])
    # emb = clf.encode_batch(batch, rel_length)
    # out_prob = clf.mods.classifier(emb).squeeze(1)
    # score, index = torch.max(out_prob, dim=-1)
    # text_lab = clf.hparams.label_encoder.decode_torch(index)
    return clf.classify_file("samples/" + audio + ".wav")
