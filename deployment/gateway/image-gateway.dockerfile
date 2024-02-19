FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["deployment/gateway/Pipfile", "deployment/gateway/Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["python/gateway.py", "python/proto.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "gateway:app"]