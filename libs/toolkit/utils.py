# -*- coding: utf-8 -*-
__author__ = "Ali Saghiran"
__copyright__ = "Copyright 2022, Resileyes Therapeutics - TraumaVoice project"
__credits__ = ["Ali Saghiran", "Older contributors"]
__version__ = "0.0.dev2"
__maintainer__ = "Ali Saghiran"
__email__ = "ali.saghiran@resileyes.com"
__status__ = "R&D"

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf


def float_to_byte(sig: np.ndarray, bitdepth: str = "int16") -> bytes:
    """float32 -> int16(PCM_16) -> byte

    Args:
        sig (np.array): _description_
        bitdepth (str, optional): _description_. Defaults to 'int16'.

    Returns:
        bytes: _description_
    """
    return float2pcm(sig, dtype=bitdepth).tobytes()  # type: ignore


def byte_to_float(byte: bytes) -> np.ndarray:
    """byte -> int16(PCM_16) -> float32"""
    return pcm2float(np.frombuffer(byte, dtype=np.int16), dtype="float32")


def pcm2float(sig: npt.ArrayLike, dtype="float32") -> np.ndarray:
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Args:
        sig (array_like): Input array, must have integral type.
        dtype (data type, optional): Desired (floating point) data type.

    Returns:
        (numpy.ndarray) Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in "iu":
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig: np.ndarray, dtype="int16") -> npt.ArrayLike:
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html

    Args:
        sig (array_like): Input array, must have floating point type.
        dtype (data type, optional): Desired (integer) data type.

    Returns:
        numpy.ndarray : Integer data, scaled and clipped to the range of the
        given *dtype*.
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != "f":
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)  # type: ignore
    if dtype.kind not in "iu":
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def audioread(filename: str) -> tuple[np.ndarray, int]:
    """Reads audio file using Matlab convention. All input audio samples are
    scaled to the range [−1; +1] and stored as 32-bit floating point numbers,
    in order to work with normalised values regardless of the actual bit-depth
    of the inputs.

    Args:
        filename (str): audio file path/name

    Returns:
        tuple[np.ndarray, int]:
            Sample rate (or sampling frequency)
            audio signal: numpy array of floats between -1 and 1, which is the
            convention for audio signal processing.
    """

    data, samplerate = librosa.load(filename, sr=None)  # type: ignore
    return (data, samplerate)


def audiowrite(
    filename: str,
    data: npt.ArrayLike,
    samplerate: int,
    subtype: str = "PCM_16",
    format: str = "wav",
):
    """_summary_

    Args:
        filename (str): Audio file name.
        data (npt.ArrayLike): Numpy array of floats between -1 and 1.
        samplerate (int): Sampling frequency (Hz).
        subtype (str, optional): _description_. Defaults to "PCM_16".
        format (str, optional): _description_. Defaults to "wav".
    """
    sf.write(filename, data, samplerate, subtype=subtype, format=format)
