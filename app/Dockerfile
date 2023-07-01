FROM python:3.9.13

WORKDIR /app

EXPOSE 8501

ENV PYTHONPATH=/app

COPY ./app/app.py /app/app.py
COPY ./app/requirements-app.txt /app/requirements-app.txt

RUN pip install -r requirements-app.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]