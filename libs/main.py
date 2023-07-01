import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from app.config import logger
from libs.features import (
    speech_to_array_fn,
    get_vad_json,
    get_signal_energy
    )

app = FastAPI()


class InputAudio(BaseModel):
    # content_format: str
    # sampling_rate: int
    content_format: str
    name: str


class Features(BaseModel):
    pitch_std: float
    loudness_mean: float
    loudness_std: float
    sound_level_db: float
    pseudo_syl_rate: float
    unvoiced_average_duration: float
    pause_average_duration: float
    pause_frequency: float
    pause_voice_ratio: float
    audio_length: str
    status: str = "unknown"


# @app.get("/")
def read_test():
    response = {"message": "Hello World"}
    logger.debug(f"Logging message: response {response}")
    return response


#@app.post("/analyse_file", response_model=Features)
def compute_audio_features(audio, sr):
    response = None
    
    logger.debug(f"signal array length: {audio.shape}")
    logger.debug(f"signal SR: {sr} Hz")
    out = get_vad_json(audio, sr)
    logger.debug(f"Timestamp vector shape: {out.shape}")

    feats = get_signal_energy(audio, sr)
    logger.debug(f"Feature dataframe shape: {feats.shape}")
    # logger.debug(f"Feature dataframe shape: {feats.columns}")
    logger.debug(f"Pseudosyllable rate /s: "
                 f"{feats['VoicedSegmentsPerSec'][0]:.2f}")
    logger.debug(f"MeanUnvoicedSegmentLength: "
                 f"{feats['MeanUnvoicedSegmentLength'][0]:.2f}")
    logger.debug(f"SoundLevel {feats['equivalentSoundLevel_dBp'][0]:.3f}")
    logger.debug(f"Loudness mean (20%, 50%, 80%): "
                 f"{feats['loudness_sma3_amean'][0]:.2f}"
                 f" ({feats['loudness_sma3_percentile20.0'][0]:.2f}"
                 f", {feats['loudness_sma3_percentile50.0'][0]:.2f}"
                 f", {feats['loudness_sma3_percentile80.0'][0]:.2f})"
                 )

    # compute pause by substracting last tmstp from the followin begin tmstp
    pauses_len = out[1:, 0] - out[:-1, 1]
    if len(pauses_len) != 0:
        mean_pause_duration = np.mean(pauses_len)
    else:
        mean_pause_duration = 0
    # n_pause per minute
    pause_freq = (pauses_len.shape[0] * sr * 60) / (audio.shape[0])
    pause_voice_ratio = np.sum(pauses_len) / out[-1, 1]
    # preprocess the image and prepare it for classification
    response = Features(
        pitch_std=round(
            float(
                feats['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'][0]
            ), 3),
        loudness_mean=round(float(feats['loudness_sma3_amean'][0]), 3),
        loudness_std=round(float(feats['loudness_sma3_stddevNorm'][0]), 3),
        sound_level_db=round(float(feats['equivalentSoundLevel_dBp'][0]), 3),
        pseudo_syl_rate=round(float(feats['VoicedSegmentsPerSec'][0]), 3),
        unvoiced_average_duration=round(
            float(feats['MeanUnvoicedSegmentLength'][0]), 3),
        pause_average_duration=round(mean_pause_duration, 3),
        pause_frequency=round(pause_freq, 3),
        pause_voice_ratio=round(pause_voice_ratio, 3),
        audio_length=(f"{np.floor(out[-1, 1] / 60)} min "
                      f"{np.ceil(out[-1, 1] % 60)} s"
                      ),
        status="success")

    # return the response as a JSON
    return response
