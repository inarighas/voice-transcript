import logging
import librosa
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from config import LogConfig
from models.speechbrain_model import transcribe
#from models.sentiment import analyse_sentiment
# from models.voxpopuli_model import transcribe

app = FastAPI()

logging.config.dictConfig(LogConfig().dict())
logger = logging.getLogger("voice-transcript")

# Some basic datastructures


class FlushedAudio(BaseModel):
    content_format: str
    encoding: str
    dtype: str
    sampling_rate: int
    content: str

class FileAudio(BaseModel):
    content_format: str
    name: str


class Transcription(BaseModel):
    content: str
    word_rate: float = 0
    sentiment: str
    sentiment_probas: str
    status: str = "unknown"


class FailedTranscription(Transcription):
    content: str = ""
    word_rate: float = 0
    sentiment: str = ""
    sentiment_probas: str = ""
    status: str = "error"


def speech_to_array_from_buffer(audio_b64: str):
    arr = base64.b64decode(audio_b64)
    speech_array = np.frombuffer(arr, dtype=np.float32)
    return speech_array

def speech_to_array_from_file(filename, target_sr):
    tmp, orig_sr = librosa.load('samples/' + filename + '.wav')
    logger.debug(f'File Loaded. Original SR: {orig_sr}, NSamples {tmp.shape}')
    if orig_sr != target_sr:
        speech_array = librosa.resample(tmp, orig_sr=orig_sr,
                                        target_sr=target_sr,
                                        res_type='kaiser_fast')
    else:
        raise ValueError("origin value is not allowed (buffer or file).")
    return speech_array, target_sr

def process_transcription(signal: np.array, sr: int):
    response = ""
    try:
        response, dur = transcribe(signal, sr)
    except ValueError:
        logger.warning("Error during transcription (see transcribe procedure).")
        raise

    logger.debug(f"Transcription: {response}")
    return response, dur


@app.get("/")
def read_test():
    """Test get request"""
    response = {"message": "Hello World"}
    logger.debug(f"Logging message: response {response}")
    return response


# Transcript post request


@app.post("/transcript_buffer", response_model=Transcription)
async def transcribe_audio_from_buffer(flushed: FlushedAudio):
    """_summary_

    Args:
        flushed (FlushedAudio): _description_

    Returns:
        _type_: _description_
    """
    response = FailedTranscription()
    audio = flushed.content

    if flushed.content_format != "audio/raw":
        logger.debug(f"Content format {flushed.content_format} unsupported.")
        response.status = "unsuppported format"

    elif flushed.encoding != "b64":
        logger.warning(f"Encoding other than `b64` is unsupported.")
        response.status = "unsupported encoding"
        return response

    elif flushed.dtype != "float32":
        logger.warning(f"dtypes other than float32 are unsupported.")
        response.status = "unsupported dtype"
        return response

    # Process by model
    arr = speech_to_array_from_buffer(audio)
    text, dur = process_transcription(arr, flushed.sampling_rate)
    sentiment = ["testing", "", "", ""]
    # analyse_sentiment(audio, flushed.sampling_rate, origin=flushed.origin)
    logger.debug(f"text: {text}")
    logger.debug(f"sentiment: {sentiment}")
    # preprocess the image and prepare it for classification
    if text != "":
        rate = len(text) * 60 / dur       # words/minute ???
        response = Transcription(
            content=text,
            sentiment=f"{sentiment[3]}",
            sentiment_probas=f"{sentiment}",
            word_rate=rate,
            status="success",
        )

    # return the response as a JSON
    return response

@app.post("/transcript_file", response_model=Transcription)
async def transcribe_audio_from_file(audio: FileAudio):
    """_summary_

    Args:
        audio (FileAudio): _description_

    Returns:
        _type_: _description_
    """
    response = FailedTranscription()
    # Ensure that the instant is okay

    if audio.content_format != "wav":
        logger.warning(f"Content format unsupported.")
        response.status = "unsuppported format"
        return response

    # Process by model
    arr, sr = speech_to_array_from_file(audio.name, 16_000)
    text, dur = process_transcription(arr, sr)
    # sentiment = analyse_sentiment(audio, flushed.sampling_rate,
    #                               origin=flushed.origin)
    sentiment = ["testing", "", "", ""]
    logger.debug(f"text: {text}")
    logger.debug(f"sentiment: {sentiment}")
    # preprocess the image and prepare it for classification
    if text != "":
        rate = len(text) * 60 / dur       # words/minute
        response = Transcription(
            content=text,
            sentiment=f"{sentiment[3]}",
            sentiment_probas=f"{sentiment}",
            word_rate=rate,
            status="success",
        )

    # return the response as a JSON
    return response
