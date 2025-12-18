import requests
import os

# 1. Konfiguration
URL = "http://localhost:8000/inpaint"
IMAGE_PATH = "imageeffd8085.jpg"  # Dein kaputtes Bild
MASK_PATH = "imageeffd8085_mask001.png"  # Die s/w Maske (Weiß = Reparieren)
OUTPUT_PATH = "result_restored.png"


def test_restoration():
    # Prüfen ob Dateien existieren
    if not os.path.exists(IMAGE_PATH) or not os.path.exists(MASK_PATH):
        print("❌ FEHLER: Bild oder Maske fehlt im Ordner!")
        return

    print(f"Sende Anfrage an {URL}...")

    # 2. Dateien zum Senden vorbereiten
    # 'image' und 'mask' müssen exakt so heißen wie im FastAPI Backend definiert
    files = {
        'image': open(IMAGE_PATH, 'rb'),
        'mask': open(MASK_PATH, 'rb')
    }

    # 3. Request senden
    try:
        response = requests.post(URL, files=files)

        # 4. Ergebnis prüfen
        if response.status_code == 200:
            # Bild speichern
            with open(OUTPUT_PATH, 'wb') as f:
                f.write(response.content)
            print(f"✅ ERFOLG! Repariertes Bild gespeichert unter: {OUTPUT_PATH}")
            print(f"Größe: {len(response.content) / 1024:.2f} KB")
        else:
            print(f"❌ SERVER FEHLER: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ VERBINDUNGSFEHLER: Läuft der Docker Container?")


if __name__ == "__main__":
    test_restoration()