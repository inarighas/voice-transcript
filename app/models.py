#import logging
import torch
import numpy as np

from speechbrain.pretrained import EncoderASR
# from config import LogConfig

# logging.config.dictConfig(LogConfig().dict())
# logger = logging.getLogger("voice-transcript")


# Initialize Speeech Recognition model
asr_model = EncoderASR.from_hparams(
    source="./pretrained_models/asr-wav2vec2-commonvoice-fr"
)

def transcribe(bytes_arr, sr):
    signal = torch.tensor(np.frombuffer(bytes_arr, dtype=np.float32))
    signal_norm = asr_model.audio_normalizer(signal, sr)
    batch = signal_norm.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    # response = str(signal.shape)
    #logger.debug(signal_norm)
    predicted_words, predicted_tokens = asr_model.transcribe_batch(batch,
                                                                   rel_length
                                                                   )
    response = str(predicted_words[0])
    # logger.debug(f"Rel_length: {rel_length}")
    # logger.debug(f"Predicted words: {predicted_words}")
    return response
