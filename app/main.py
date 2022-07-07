import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import logger
from app.models.speechbrain_model import transcribe
from app.utils import (
    compute_word_rate,
    speech_to_array_from_buffer,
    speech_to_array_from_file,
)

# from models.sentiment import analyse_sentiment
# from models.voxpopuli_model import transcribe

app = FastAPI()

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


def process_transcription(signal: np.array, sr: int):
    response = ""
    try:
        response, dur = transcribe(signal, sr)
    except ValueError:
        logger.warning(
            "Error during transcription, see transcribe procedure."
        )
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


@app.post("/transcribe_buffer", response_model=Transcription)
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
        logger.warning(
            f"Content format {flushed.content_format} unsupported."
        )
        response.status = "unsuppported format"
        raise HTTPException(
            status_code=404, detail="Unsuppported format"
        )

    elif flushed.encoding != "b64":
        logger.warning("Encoding other than `b64` is unsupported.")
        raise HTTPException(
            status_code=404, detail="Unsuppported encoding"
        )

    elif flushed.dtype != "float32":
        logger.warning("dtypes other than float32 are unsupported.")
        raise HTTPException(
            status_code=404, detail="Unsuppported dtype"
        )

    # Process by model
    arr = speech_to_array_from_buffer(audio)
    text, dur = process_transcription(arr, flushed.sampling_rate)
    sentiment = ["testing", "", "", ""]
    # analyse_sentiment(audio, flushed.sampling_rate, origin=flushed.origin)
    logger.debug(f"text: {text}")
    logger.debug(f"sentiment: {sentiment}")
    # preprocess the image and prepare it for classification
    if text != "":
        rate = compute_word_rate(text, dur)
        response = Transcription(
            content=text,
            sentiment=f"{sentiment[3]}",
            sentiment_probas=f"{sentiment}",
            word_rate=rate,
            status="success",
        )

    # return the response as a JSON
    return response


@app.post("/transcribe_file", response_model=Transcription)
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
        logger.warning("Requested content format is unsupported.")
        raise HTTPException(
            status_code=404, detail="Unsupported format"
        )

    # Process by model
    arr, sr = speech_to_array_from_file(
        audio.name, 16_000, audio.content_format
    )
    text, dur = process_transcription(arr, sr)
    # sentiment = analyse_sentiment(audio, flushed.sampling_rate,
    #                               origin=flushed.origin)
    sentiment = ["testing", "", "", ""]
    logger.debug(f"text: {text}")
    logger.debug(f"sentiment: {sentiment}")
    # preprocess the image and prepare it for classification
    if text != "":
        rate = compute_word_rate(text, dur)
        response = Transcription(
            content=text,
            sentiment=f"{sentiment[3]}",
            sentiment_probas=f"{sentiment}",
            word_rate=rate,
            status="success",
        )

    # return the response as a JSON
    return response
