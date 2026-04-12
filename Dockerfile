FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY inference.py .
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]