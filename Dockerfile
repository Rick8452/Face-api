
# Imagen base con conda (multi-arch: arm64/amd64)
FROM conda/c3i-linux-64:latest
# Evita prompts interactivos en conda
ENV CONDA_ALWAYS_YES=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Crea carpeta de la app
WORKDIR /app
# Copia y crea el entorno conda
COPY environment.yml /app/
RUN conda env create -f /app/environment.yml && conda clean -afy
# Asegura que el PATH use el entorno "fr"
SHELL ["bash", "-lc"]
#RUN echo "conda activate fr" >> ~/.bashrc
ENV PATH /opt/conda/envs/fr/bin:$PATH
# Copia tu app
COPY . /app
# Crea carpeta de datos (persistiremos por volumen)
RUN mkdir -p /app/data/users
# Exponer puerto
EXPOSE 8000
# Comando de arranque (uvicorn usando el entorno "fr")
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port 8000"]