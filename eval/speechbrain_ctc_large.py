import torch
import librosa

from datasets import load_dataset
from transformers import MCTCTForCTC, MCTCTProcessor

model = MCTCTForCTC.from_pretrained("speechbrain/m-ctc-t-large")
processor = MCTCTProcessor.from_pretrained("speechbrain/m-ctc-t-large")

# load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy",
#                   "clean",
#                   split="validation")

# # feature extraction
# input_features = processor(ds[0]["audio"]["array"],
#                            sampling_rate=ds[0]["audio"]["sampling_rate"],
#                            return_tensors="pt").input_features

filename = "bonjour"
ext = "wav"
tmp, orig_sr = librosa.load("samples/" + filename + "." + ext)
input_features = processor(tmp, sampling_rate=orig_sr,
                           return_tensors="pt").input_features

# retrieve logits
with torch.no_grad():
    logits = model(input_features).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)