# first stage
FROM python:3.8-slim AS builder
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip uninstall -y keras

# second stage
FROM python:3.8-slim AS image

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY . .
EXPOSE 8080
CMD [ "python", "fapi.py" ]
