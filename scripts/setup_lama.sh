#!/bin/bash
set -e # Bricht ab, wenn ein Fehler passiert

echo "--- Starte LaMa Model Setup ---"

# 1. Sicherheits-Check: Sind wir im Projekt-Root?
if [ ! -d "lama" ]; then
    echo "[FEHLER] Ordner 'lama' nicht gefunden!"
    echo "Bitte führe dieses Skript im Ordner aus, in dem auch dein 'lama' Repo liegt."
    exit 1
fi

if [ -d "big-lama" ]; then
    echo "[INFO] 'big-lama' Ordner existiert bereits. Überspringe Download."
    exit 0
fi

# 2. Download des Modells (Big-Lama)
# Wir nutzen den direkten Link aus der Anleitung (HuggingFace Mirror, da Yandex oft down ist)
echo "[1/3] Lade big-lama.zip herunter..."
curl -L -o big-lama.zip https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip

# 3. Entpacken
echo "[2/3] Entpacke Archiv..."
unzip -q big-lama.zip

# 4. Aufräumen & Verifizieren
echo "[3/3] Bereinige temporäre Dateien..."
rm big-lama.zip

# Prüfung auf die kritische Datei
if [ -f "big-lama/models/best.ckpt" ]; then
    echo "------------------------------------------------"
    echo "✅ ERFOLG! Setup abgeschlossen."
    echo "Das Modell liegt bereit unter: ./big-lama/models/best.ckpt"
    echo "Du kannst jetzt den Docker Container bauen."
    echo "------------------------------------------------"
else
    echo "❌ FEHLER: Die Datei 'best.ckpt' wurde nicht gefunden."
    exit 1
fi