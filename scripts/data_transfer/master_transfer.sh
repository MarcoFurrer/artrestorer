#!/bin/bash

# === DEIN BUCKET NAME (Hier anpassen!) ===
export BUCKET_URL="gs://artrestorer"

# === SETUP (Da neue VM) ===
echo "--- STARTE INSTALLATION ---"
sudo apt-get update -qq
sudo apt-get install unzip python3-pip -y -qq
pip3 install kaggle --break-system-packages
export PATH=$PATH:~/.local/bin

echo "--- START TRANSFER PIPELINE ---"
echo "Zeitstempel: $(date)"

# Funktion für den wiederholbaren Prozess
process_file () {
    local FILE_NAME=$1
    local TARGET_FOLDER=$2 # z.B. "train" oder "test"

    echo "##################################################"
    echo " VERARBEITE: $FILE_NAME -> gs://$TARGET_FOLDER"
    echo "##################################################"

    # 1. Download
    echo "[1/4] Download..."
    if kaggle competitions download -c painter-by-numbers -f $FILE_NAME; then
        echo "Download OK."
    else
        echo "FEHLER: Download fehlgeschlagen. Abbruch für diese Datei."
        return 1
    fi

    # 2. Entpacken
    echo "[2/4] Entpacken..."
    # -o überschreibt ohne zu fragen, -q für leise
    unzip -q -o $FILE_NAME -d temp_process_folder

    # 3. Upload
    echo "[3/4] Upload in Bucket..."
    gsutil -m cp -r temp_process_folder/* $BUCKET_URL/$TARGET_FOLDER/

    # 4. Aufräumen
    echo "[4/4] Aufräumen..."
    rm $FILE_NAME
    rm -rf temp_process_folder
    echo "Fertig mit $FILE_NAME"
}

# === SCHRITT 1: TEST DATA (11.9 GB) ===
# Wir machen das zuerst, wie du wolltest
process_file "test.zip" "test"

# === SCHRITT 2: TRAIN DATA (Teil 1 bis 9) ===
for i in {1..9}
do
    process_file "train_$i.zip" "train"
done

echo "##################################################"
echo " ALLE AUFGABEN ERLEDIGT! VM KANN GELÖSCHT WERDEN."
echo "##################################################"