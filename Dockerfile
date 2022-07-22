FROM python:3.10-slim AS base

from base as builder
RUN mkdir /install
COPY  ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /requirements.txt

FROM base
RUN apt-get update -y && apt-get install -y --no-install-recommends libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY  ./requirements.txt /requirements.txt
RUN mkdir /code && mkdir /code/pretrained_models
COPY ./app /code/app
COPY  --chown=1001:1001 ./samples        /code/samples
COPY  --chown=1001:1001 ./pretrained_models/EncoderASR_temp             /code/pretrained_models/EncoderASR_temp

RUN useradd -ms /bin/bash 1001
RUN mkdir /tmp/numba_cache
RUN chmod 777 /tmp/numba_cache 
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
EXPOSE 8001
USER 1001

WORKDIR "/code"
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
# ENTRYPOINT ["gunicorn","-t","180","-w","2","-k","uvicorn.workers.UvicornWorker","-b","0.0.0.0:8000","app.main:app"]
