# `voice-transcript` app & microservice

Miniapp and corresponding microservice implementation performing audio/voice transcription (speech-to-text) based on on speechbrain/hugging face deep model (wav2vec2).

## Content

**Input**:

- b64 encoded audio (`buffer` input) or a file path (`file` input).

**Output**:

- text transcription of input audio (french);
- Speech rate in word per minute;
- Emotion related feature computing;

## Project structure

- `app/` contains application source code.
- `eval/` gathers some evaluation/exploratory scripts and jupyter notebooks.
- `tests/` is the testing files folder.
- `samples/` contains some audio files for testing.
- `libs/` contains mainly some audio processing (feature extractions) helpers
- `demo/` contains a streamlit interface that runs the service internaly and shows result on a webpage.



## Running the demo

- Make sure all dependencies are installed (see Dependencies section),
- Run `streamlit demo/interactive_app.py`,
- Open in your favourite browser (tested on firefox and Chromium) the generated (localhost) link if not opened automatically.

## Runnig the service

### **Manual installation of the pretrained models**

The pretrained models are not in the github. Currently they should be downloaded and put in the folder `app/pretrained_models`.
Two models are needed:

- Speechbrain's `wav2vec2-FR-7K-large/`: download using

    ```{bash}
    git lfs clone https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large
    ```

- Lebenchmark's `asr-wav2vec2-commonvoice-fr/`: download using

    ```{bash}
    git lfs clone https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-fr
    ```

### 1. **Locally & Manually**

First, create a python virtual environment and check if all dependencies are properly installed.

```{bash}
python -m venv ./.venv && source ./.venv/bin/activate
pip install -r requirements.txt
```

To run the service, just run the file `run.py` with python:

```{bash}
python run.py
```

In this file, the only thing which is done is the `uvicorn` command to run the service.
The api communicate with the following ip adress `127.0.0.1`  using port `8000`.

### 2. **Using Docker**

Be sure to get all the pretrained models installed in their corresponding folders. This should be done manually or added to any automatic deployement of the micro service.

Building the image:

```{bash}
docker build -t voice-transcript .
```

If no Docker network is established, create network to assign static IP address to the service:

```{bash}
docker network create --subnet=172.20.0.0/16 demo-network
```

Running the image:

```{bash}
docker run -d --net demo-network --ip 172.20.0.11 -p 8001:8001  -v path/to/local/pretrained_models/folder:/path/to/container/pretrained_models/folder voice-transcript
```

> **_NOTE:_**  Currently the command is `docker run -d --net demo-network --ip 172.20.0.11 -p 8001:8001  -v /root/voice-transcript/pretrained_models:/code/pretrained_models voice-transcript`

In this setting, the api communicate with the following ip adress `172.20.0.11`  using port `8001`.

### **Communicate with the API using CURL**

- Test GET request

```{bash}
curl --request GET \
  --url http://172.20.0.11:8001/ \
  --header 'Content-Type: application/json'
```

- Transcribe `bonjour.wav` (POST request)

```{bash}
curl -X 'POST' \
  'http://172.20.0.11:8001/transcribe_file' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "content_format": "wav",
  "name": "bonjour"
}'
```

## Dependencies

*to run the API and the streamlit demo*.

- see the full dependencies available in `requirements.txt`


## Credits

**Author**: Ali Saghiran
