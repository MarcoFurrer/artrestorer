import sys
import os
import torch
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from io import BytesIO
from PIL import Image

# --- Setup LaMa Pfade ---
sys.path.append(os.path.join(os.getcwd(), 'lama'))

# LaMa Imports
from saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf

app = FastAPI(title="LaMa Inpainting API")

# --- Globale Variablen ---
MODEL = None
DEVICE = "cpu"  # Setze auf "cuda" für GPU Support
MODEL_PATH = "big-lama/models/final_model_custom_hydra_logs_models_last.ckpt"
CONFIG_PATH = "big-lama/final_model_custom_hydra_logs_.hydra_config.yaml"


def load_lama_model():
    print("Lade LaMa Konfiguration...")
    train_config = OmegaConf.load(CONFIG_PATH)
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    print(f"Lade LaMa Checkpoint von {MODEL_PATH}...")
    model = load_checkpoint(train_config, MODEL_PATH, strict=False, map_location=DEVICE)
    model.freeze()
    model.to(DEVICE)
    print("Modell erfolgreich geladen!")
    return model


@app.on_event("startup")
async def startup_event():
    global MODEL
    MODEL = load_lama_model()


def pad_img_to_modulo(img, mod=8):
    """
    Fügt Padding hinzu, damit Höhe und Breite durch 8 teilbar sind.
    """
    if len(img.shape) == 2:  # Graustufen
        h, w = img.shape
    else:  # RGB
        h, w = img.shape[:2]

    new_h = (h + mod - 1) // mod * mod
    new_w = (w + mod - 1) // mod * mod

    pad_h = new_h - h
    pad_w = new_w - w

    if len(img.shape) == 2:
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    return img_padded, h, w


@app.post("/inpaint")
async def predict(image: UploadFile = File(...), mask: UploadFile = File(...)):
    if MODEL is None:
        return {"error": "Modell nicht geladen"}

    # 1. Daten lesen
    img_bytes = await image.read()
    mask_bytes = await mask.read()

    img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    mask_pil = Image.open(BytesIO(mask_bytes)).convert("L")

    img_np = np.array(img_pil)
    mask_np = np.array(mask_pil)

    # 2. Padding (Crash-Fix)
    img_padded, orig_h, orig_w = pad_img_to_modulo(img_np, mod=8)
    mask_padded, _, _ = pad_img_to_modulo(mask_np, mod=8)

    # 3. Normalisieren & Tensor
    img_t = img_padded.astype('float32') / 255.0
    mask_t = mask_padded.astype('float32') / 255.0
    mask_t = (mask_t > 0.5).astype('float32')

    img_t = torch.from_numpy(img_t).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    mask_t = torch.from_numpy(mask_t).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 4. Inferenz
    with torch.no_grad():
        batch = {'image': img_t, 'mask': mask_t}
        result = MODEL(batch)  # Ergebnis ist ein Dict

        # --- FIX HIER: Zugriff auf Key 'inpainted' ---
        # result['inpainted'] hat Shape [Batch, Channel, Height, Width]
        batch_out = result['inpainted']

        # [0] nimmt das erste Bild im Batch, dann Permute zu [H, W, C]
        cur_res = batch_out[0].permute(1, 2, 0).detach().cpu().numpy()

        # 5. Unpadding
        cur_res = cur_res[:orig_h, :orig_w, :]

        # Zurück zu 0-255 uint8
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)

    # 6. Als PNG senden
    success, encoded_image = cv2.imencode('.png', cur_res)
    return Response(content=encoded_image.tobytes(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)