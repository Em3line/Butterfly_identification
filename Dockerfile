
FROM python:3.8.6-buster

COPY app /app
COPY requirements.txt /requirements.txt
COPY raw_data/Docker /raw_data/Docker

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn API.api:app --host 0.0.0.0 --port $PORT

#to run the container : docker run -p 8080:8000 api
