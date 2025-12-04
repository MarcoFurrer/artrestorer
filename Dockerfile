# Basis: Python 3.8 (Zwingend für LaMa)
FROM python:3.8-slim

# System-Abhängigkeiten für OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Installiere Abhängigkeiten (Layer Caching nutzen)
# FIX: albumentations==0.5.2 hinzugefügt
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
    scikit-learn

# Kopiere den gesamten Kontext (lama repo, big-lama weights, app.py)
COPY . /app

# Umgebungsvariable für Python Pfad setzen
ENV PYTHONPATH="${PYTHONPATH}:/app/lama"

# Port freigeben
EXPOSE 8000

# Startbefehl
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]