# Basis: Python 3.8 (Zwingend für LaMa)
FROM python:3.8-slim

# 1. NEU: System-Updates inkl. Google Cloud SDK (gsutil)
# Wir installieren hier 'google-cloud-cli', damit wir 'gsutil' für High-Speed Downloads haben.
# libgl1/libglib sind weiterhin für OpenCV nötig.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Installiere Abhängigkeiten (Layer Caching nutzen)
# FIX: albumentations==0.5.2 ist drin
# NEU: google-cloud-storage hinzugefügt
RUN pip install --no-cache-dir \
    torch==1.8.0 \
    torchvision==0.9.0 \
    pytorch-lightning==1.2.9 \
    hydra-core==1.1.0 \
    scikit-image \
    opencv-python \
    fastapi \
    uvicorn \
    python-multipart \
    pyyaml \
    easydict \
    webdataset \
    kornia==0.5.0 \
    albumentations==0.5.2 \
    pandas \
    scikit-learn \
    google-cloud-storage

# Kopiere den gesamten Kontext (lama repo, big-lama weights, app.py)
COPY . /app

# Umgebungsvariable für Python Pfad setzen
ENV PYTHONPATH="${PYTHONPATH}:/app/lama"

# Port freigeben
EXPOSE 8000

# Startbefehl (wird beim Training via Command überschrieben)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]