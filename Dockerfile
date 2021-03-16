FROM nvcr.io/nvidia/pytorch:21.02-py3

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

COPY . /app/
WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install \
  git+https://github.com/1adrianb/face-alignment \
  -r requirements.txt
