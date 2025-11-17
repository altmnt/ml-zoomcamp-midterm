FROM python:3.12-slim

RUN pip install uv

WORKDIR /code

COPY "pet_preference_model.bin" "index.html" "pyproject.toml" ".python-version" "uv.lock" "./"

RUN uv sync --locked

COPY "serve.py" "./"
EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]
