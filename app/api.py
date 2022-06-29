import logging
from fastapi import FastAPI
from pydantic import BaseModel

from config import LogConfig
from models.speechbrain_model import transcribe
# from models.voxpopuli_model import transcribe

app = FastAPI()

logging.config.dictConfig(LogConfig().dict())
logger = logging.getLogger("voice-transcript")

# Some basic datastructures


class Instant(BaseModel):
    user_id: str
    timestamp: str | dict  # str: array[begin, end]


class FlushedAudio(Instant):
    content_format: str
    encoding: str
    dtype: str
    sampling_rate: int
    content: str
    origin: str


class Transcription(Instant):
    content: str
    status: str = "unknown"
    rate: float = 0


class FailedTranscription(Transcription):
    content: str = ""
    status: str = "error"


def process_transcription(audio: str, sr: int, encoding: str, origin="buffer"):
    response = ""
    try:
        response, dur = transcribe(audio, sr, origin=origin)
    except ValueError:
        response = ""
        logger.debug("Corrupt audio input.")
        raise

    logger.debug(f"Transcription: {response}")
    return response, dur


# Test get request


@app.get("/")
def read_test():
    response = {"message": "Hello World"}
    logger.debug(f"Logging message: response {response}")
    return response


# Transcript post request


@app.post("/transcript", response_model=Transcription)
async def transcript_audio(flushed: FlushedAudio):
    response = FailedTranscription(user_id=flushed.user_id,
                                   timestamp=flushed.timestamp)
    # Ensure that the instant is okay
    audio = flushed.content

    if flushed.content_format != "audio/raw":
        logger.debug(f"Content format {flushed.content_format} unsupported.")
        response.status = "unsuppported format"

    elif flushed.encoding != "b64":
        response.status = "unsupported encoding"
        logger.debug(f"Encoding {flushed.encoding} unsupported.")
        return response

    elif flushed.dtype != "float32":
        logger.debug(f"{flushed.dtype} dtype unsupported.")
        response.status = "unsupported dtype"
        return response

    # Process by model
    else:
        text, dur = process_transcription(audio,
                                          flushed.sampling_rate,
                                          flushed.encoding,
                                          origin=flushed.origin)
        logger.debug(f"text: {text}")
        # preprocess the image and prepare it for classification
        if text != "":
            rate = len(text) * 60 / dur       # words/minute
            response = Transcription(
                user_id=flushed.user_id,
                timestamp=flushed.timestamp,
                content=text,
                rate=rate,
                status="success",
            )

    # return the response as a JSON
    return response
