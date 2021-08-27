FROM python:3.7-slim-buster AS builder
RUN apt-get update && apt-get install make
RUN apt-get install libgomp1

FROM builder AS builder-venv

ADD . /demo
RUN pip install --disable-pip-version-check -r app/requirements.txt

WORKDIR /demo