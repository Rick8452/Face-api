# Imagen base de Python
FROM python:3.11-slim

# Desactiva bytecode + buffer de logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Crea directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema si las necesitas
# (por ejemplo, para psycopg2, PIL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Copia el requirements y lo instala con pip
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia tu código
COPY . /app

# Crea carpeta persistente
RUN mkdir -p /app/data/users

# Exponer puerto
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]