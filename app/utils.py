import base64

import librosa
import numpy as np
from fastapi import HTTPException

from app.config import logger


def compute_word_rate(text: str, duration: float) -> float:
    """_summary_

    Args:
        text (str): transcribed text
        duration (float): in seconds

    Returns:
        float: word rate in number of words per minute.
    """
    return len(text.split(" ")) * 60 / duration


def speech_to_array_from_buffer(audio_b64: str):
    arr = base64.b64decode(audio_b64)
    speech_array = np.frombuffer(arr, dtype=np.float32)
    return speech_array


def speech_to_array_from_file(filename, target_sr, ext):
    try:
        tmp, orig_sr = librosa.load("samples/" + filename + "." + ext)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except:
        raise HTTPException(status_code=404, detail="File error")

    logger.debug(
        f"File Loaded. Original SR: {orig_sr}, NSamples {tmp.shape}"
    )
    if orig_sr != target_sr:
        speech_array = librosa.resample(
            tmp,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type="kaiser_fast",
        )
    return speech_array, target_sr
