FROM python:3.10-slim AS base
ENV PYTHONUNBUFFERED 1

from base as builder

RUN mkdir /code
RUN mkdir /code/app
RUN mkdir /code/pretrained_models


FROM base
COPY ./app /code/app
COPY  --chown=1001:1001 ./samples        /code/samples
COPY  --chown=1001:1001 ./pretrained_models/asr-wav2vec2-commonvoice-fr /code/pretrained_models/asr-wav2vec2-commonvoice-fr
COPY  --chown=1001:1001 ./pretrained_models/EncoderASR_temp             /code/pretrained_models/EncoderASR_temp
COPY  --chown=1001:1001 ./pretrained_models/wav2vec2-FR-7K-large        /code/pretrained_models/wav2vec2-FR-7K-large
COPY  ./requirements.txt                                                /code/requirements.txt

RUN useradd -ms /bin/bash 1001
RUN apt-get update -y && apt-get install -y --no-install-recommends libsndfile1
RUN pip install --no-cache-dir --upgrade -r code/requirements.txt
RUN mkdir /tmp/numba_cache
RUN chmod 777 /tmp/numba_cache 
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
EXPOSE 8000
USER 1001


WORKDIR "/code"
# ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
ENTRYPOINT ["gunicorn","-t","180","-w","2","-k","uvicorn.workers.UvicornWorker","-b","0.0.0.0:8000","app.main:app"]
