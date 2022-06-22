import base64
import logging

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from speechbrain.pretrained import EncoderASR

from config import LogConfig

app = FastAPI()

logging.config.dictConfig(LogConfig().dict())
logger = logging.getLogger("voice-transcript")

# Initialize Speeech Recognition model
asr_model = EncoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-commonvoice-fr",
    savedir="pretrained_models/asr-wav2vec2-commonvoice-fr",
)


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


class Transcription(Instant):
    content: str
    status: str = "unknown"


class FailedTranscription(Transcription):
    content: str = ""
    status: str = "error"


def process_transcription(audio: str, sr: int, encoding: str) -> str:
    response = ""
    bytes_arr = base64.b64decode(audio)
    try:
        signal = torch.tensor(np.frombuffer(bytes_arr, dtype=np.float32))
        signal_norm = asr_model.audio_normalizer(signal, sr)
        batch = signal_norm.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        # response = str(signal.shape)
        logger.debug(signal_norm)
        predicted_words, predicted_tokens = asr_model.transcribe_batch(
            batch, rel_length
        )
        response = str(predicted_words[0])
        logger.debug(f"Rel_length: {rel_length}")
        logger.debug(f"Predicted words: {predicted_words}")
    except ValueError:
        response = ""
        logger.debug("Corrupt audio input.")

    logger.info(f"Transcription: {response}")
    return response


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
        text = process_transcription(audio,
                                     flushed.sampling_rate,
                                     flushed.encoding)
        # preprocess the image and prepare it for classification
        if text != "":
            response = Transcription(
                user_id=flushed.user_id,
                timestamp=flushed.timestamp,
                content=text,
                status="success",
            )

    # return the response as a JSON
    return response
