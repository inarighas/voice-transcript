# voice-transcript

Microservice performing audio/voice transcription (speech-to-text).


## Content 

**Input**: b64 encoded audio or reference in S3 bucket.
**Output**: character string representing text transcription.


## Project structure

- `app/` contains application source code.
- `eval/` gathers some evaluation and exploratory scripts.
- `tests/` is the testing files folder.
- `samples/` contains some audio files for testing.


## Dependencies

FastAPI, numpy, pydantic, speechbrain, torch


## Credits

- Developed by Ali Saghiran for Resileyes Therapeutics.
