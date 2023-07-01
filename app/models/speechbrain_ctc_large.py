import numpy as np
import torch
from speechbrain.pretrained import EncoderASR, EncoderDecoderASR

from app.config import logger

import torch
import torchaudio
from datasets import load_dataset
from transformers import MCTCTForCTC, MCTCTProcessor

LANG_ID = "fr"
MODEL_PATH = "./pretrained_models/asr-wav2vec2-commonvoice-fr"
TARGET_SR = 8_000

# MODEL_PATH = "./pretrained_models/asr-crdnn-commonvoice-fr"
# TARGET_SR = 16_000

# Initialize Speeech Recognition model
# CommonVoice6.1-Fr  WER 9.96 (self-reported)
asr_model = EncoderASR.from_hparams(source=MODEL_PATH)
# asr_model = EncoderDecoderASR.from_hparams(source=MODEL_PATH)

model = MCTCTForCTC.from_pretrained("speechbrain/m-ctc-t-large")
processor = MCTCTProcessor.from_pretrained("speechbrain/m-ctc-t-large")

def transcribe(speech_arr: np.array, sr):
    duration = len(speech_arr) / sr
    signal = torch.tensor(speech_arr)
    signal_norm = asr_model.audio_normalizer(signal, sr)
    batch = signal_norm.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    # response = str(signal.shape)
    # logger.debug(signal_norm)
    logger.debug(f"Rel_length: {rel_length}")
    logger.debug(f"Signal: {batch}")
    predicted_words, predicted_tokens = asr_model.transcribe_batch(
        batch, rel_length
    )
    logger.debug(f"Predicted words: {predicted_words}")
    return predicted_words[0].lower(), duration






 # load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
 
# feature extraction
# input_features = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt").input_features 


# retrieve logits
with torch.no_grad():
    logits = model(input_features).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)