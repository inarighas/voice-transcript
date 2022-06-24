FROM python:3.10-slim

WORKDIR code/

COPY app/  /code/app/

COPY ./model_checkpoints/ /code/model_checkpoints/

COPY ./pretrained_models/ /code/pretrained_models/

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["tree", "."]

CMD ["python", "app/run.py"]