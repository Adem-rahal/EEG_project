FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libgtk2.0-dev git 
RUN apt-get install wget

ADD . /QBCI/

WORKDIR /QBCI

RUN pip install -r requirements.txt
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN wget "https://hkinsley.com/static/downloads/bci/model_data_v2.7z"

