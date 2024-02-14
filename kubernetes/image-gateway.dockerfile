FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["kubernetes/Pipfile", "kubernetes/Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["kubernetes/gateway.py", "kubernetes/proto.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "gateway:app"]