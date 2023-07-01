import librosa
import base64
import numpy as np
import opensmile

from typing import cast
from json import JSONEncoder
from libs.config import logger
# from pystoi import stoi
from libs.toolkit.features.vad import VAD
from libs.toolkit.features.feature_extractor import (
    DefaultOpenSmileFeatureExtractor as OSExtractor
)
# from traumavoice.features.noise_reduc import NoiseReducer
# from traumavoice.features.wav_pipeline import WavPipeline
# import ..traumavoice.utils as utl


def speech_to_array_fn(audio, origin="file"):
    if origin == "buffer":
        arr = base64.b64decode(audio)
        tmp = np.frombuffer(arr, dtype=np.float32)
        sr = 0
    elif origin == "file":
        tmp, sr = librosa.load('samples/' + audio + '.wav')
        # logger.debug(f'File Loaded. Original SR: {orig_sr},
        # NSamples {tmp.shape}')
        # speech_array = librosa.resample(tmp, orig_sr=orig_sr,
        #                                 target_sr=TARGET_SR)
    else:
        raise(ValueError("origin value must be either file or buffer"))
    return tmp, sr


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_vad_json(signal, sr):

    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if sr not in [8_000, 16_000, 32_000, 48_000]:
        logger.debug(
            f"Resampling to 16k Hz.\n"
            f"\t\tInput sr ({sr}Hz) is not "
            f"in the accepted range (8, 16, 32 or 48 kHz)."
        )
        signal = librosa.resample(signal, orig_sr=sr,
                                  target_sr=16_000,
                                  res_type='kaiser_fast')
        new_sr = 16_000
    else:
        new_sr = sr

    # They're the default parameters, only writing them here as an example
    # params = {"aggressivity": 2, "vad.json": True}
    # denoiser = NoiseReducer('', None, stationary=False)
    vad = VAD('', None, aggressivity=2, json=True)

    # extractor_bis = OSExtractor('', 'eGeMAPSv02', None,
    # feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas)
    transformed = vad.transform(signal, new_sr, timestamps=True)
    out_signal, out_sr, tmstmp = cast(tuple[np.ndarray, int, list],
                                      transformed)
    return np.array(tmstmp)


def get_signal_energy(signal, sr):
    extractor_bis = OSExtractor('', 'eGeMAPSv02', None,
                                feature_level=(
                                    opensmile.FeatureLevel.Functionals
                                )
                                )
    df = extractor_bis.transform(signal, sr)[0]
    return df
