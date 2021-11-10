FROM ubuntu
RUN apt update
RUN apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-cpu pillow fastapi uvicorn jinja2 python-multipart
RUN pip3 uninstall -y keras
COPY . /app
WORKDIR /app
CMD python3 fapi.py
