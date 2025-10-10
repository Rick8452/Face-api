FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependencias del sistema necesarias (ajusta según tu proyecto)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Copia requirements.txt y instala paquetes Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia la app
COPY . /app

# Carpeta para datos persistentes
RUN mkdir -p /app/data/users

EXPOSE 8000

# Arranca uvicorn directamente
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]