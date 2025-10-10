FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias de sistema necesarias para OpenCV/dlib en runtime (y para compilar si hiciera falta)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1 || apt-get install -y --no-install-recommends libgl1-mesa-glx; \
    apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Instala deps en un venv
COPY requirements.txt /app/requirements.txt
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Copia el código
COPY . /app

# Carpeta de datos
RUN mkdir -p /app/data/users /app/data/ocr

EXPOSE 8000

# Usa el Python del venv para lanzar uvicorn
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]