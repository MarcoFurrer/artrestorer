#!/bin/bash

# === DEIN BUCKET ===
export BUCKET_URL="gs://artrestorer"

echo "--- SYSTEM VORBEREITUNG ---"
# 1. Installiere 7zip (p7zip-full) statt nur unzip
sudo apt-get update -qq
sudo apt-get install p7zip-full python3-pip -y -qq
pip3 install kaggle --break-system-packages
export PATH=$PATH:~/.local/bin

# 2. SWAP ERZWINGEN (Sicherheitspuffer auch bei 32GB RAM)
# Wir legen 8GB Swap an, falls der RAM kurz voll läuft
if ! grep -q "partition" /proc/swaps; then
    echo "Erstelle Safety-Swap..."
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
fi

# Funktion für den Prozess
process_file () {
    local FILE_NAME=$1
    local TARGET_FOLDER=$2

    echo "--------------------------------------------------"
    echo " START: $FILE_NAME -> $TARGET_FOLDER"
    echo "--------------------------------------------------"

    # Download
    if kaggle competitions download -c painter-by-numbers -f $FILE_NAME; then
        echo "Download OK."
    else
        echo "FEHLER beim Download. Abbruch."
        return 1
    fi

    # Entpacken mit 7zip (Effizienter als unzip)
    # x = eXtract with full paths
    # -y = yes to all prompts
    # -o... = output directory (kein Leerschlag nach -o!)
    echo "Entpacke mit 7zip..."
    7z x $FILE_NAME -o"temp_folder" -y > /dev/null

    # Upload
    echo "Upload in Cloud Storage..."
    gsutil -m cp -r temp_folder/* $BUCKET_URL/$TARGET_FOLDER/

    # Aufräumen
    echo "Aufräumen..."
    rm $FILE_NAME
    rm -rf temp_folder
    echo "FERTIG mit $FILE_NAME"
}

# === ABLAUF ===

# 1. Das problematische Test-File (11.9 GB)
process_file "test.zip" "test"

# 2. Die Train-Files (Train 1-9)
for i in {1..9}
do
    process_file "train_$i.zip" "train"
done

echo "ALLES ERLDIGT. VM KANN GELÖSCHT WERDEN."