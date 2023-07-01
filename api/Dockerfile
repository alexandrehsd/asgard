FROM python:3.9.13

WORKDIR /api

EXPOSE 8000

ENV PYTHONPATH=/api

COPY ./api/requirements-api.txt /api/requirements-api.txt
COPY ./api/utils.py /api/utils.py
COPY ./api/api.py /api/api.py

RUN mkdir ./api
COPY ./api/model /api/api/model

COPY asgard asgard

RUN pip install -r requirements-api.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]