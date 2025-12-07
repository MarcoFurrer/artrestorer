import os
import sys
import subprocess
import argparse
import time
import shutil
import traceback
import cv2
import numpy as np

# --- PFADE ---
LOCAL_DATA_ROOT = "/tmp/dataset"
LOCAL_TRAIN_DIR = os.path.join(LOCAL_DATA_ROOT, "train")
LOCAL_VAL_DIR = os.path.join(LOCAL_DATA_ROOT, "visual_test")
LOCAL_REAL_VAL_DIR = os.path.join(LOCAL_DATA_ROOT, "val")
LOCAL_MODEL_DIR = "/tmp/experiments"
PRETRAINED_CKPT = "/app/big-lama/models/best.ckpt"

# NEU: Eigener Output-Ordner für Phase 2
OUTPUT_FOLDER_NAME = "final_model_custom"


def run_cmd(cmd):
    """Führt Befehl laut aus"""
    print(f"Executing: {cmd}")
    subprocess.check_call(cmd, shell=True)


def run_cmd_silent(cmd):
    """Führt Befehl KOMPLETT LEISE aus."""
    print(f"Executing (silent): {cmd}")
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print(f"⚠️ FEHLER beim stillen Ausführen von: {cmd}")
        print("🔁 Wiederhole Befehl laut zur Diagnose...")
        subprocess.check_call(cmd, shell=True)


def flatten_directory(directory):
    """Holt Dateien aus Unterordnern hoch (rekursiv)."""
    print(f"🔨 Flattening {directory} ...")
    count = 0
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            source = os.path.join(root, file)
            target = os.path.join(directory, file)
            if root != directory:
                try:
                    shutil.move(source, target)
                    count += 1
                except:
                    pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except:
                pass
    print(f"✅ {count} Dateien verschoben.")


def resize_images_in_dir(directory, size=(512, 512)):
    """
    NEU: Skaliert alle Bilder in einem Ordner auf eine feste Größe.
    Notwendig für Batching (batch_size > 1).
    """
    print(f"📏 Resizing Bilder in {directory} auf {size}...")
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    count = 0
    for f in files:
        path = os.path.join(directory, f)
        try:
            img = cv2.imread(path)
            if img is None: continue
            # Resize
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            # Überschreiben
            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            count += 1
        except:
            pass
    print(f"✅ {count} Bilder resized.")


def process_and_merge_masks(val_dir, mask_source_dir):
    """
    Nimmt Masken, RESIZED sie auf 512x512, konvertiert sie zu PNG
    und speichert sie als _mask.png.
    """
    print(f"⚙️ Verarbeite & Resize Masken von {mask_source_dir} nach {val_dir}...")

    mask_files = os.listdir(mask_source_dir)
    total = len(mask_files)
    count = 0

    for i, f in enumerate(mask_files):
        src_path = os.path.join(mask_source_dir, f)
        base_name = os.path.splitext(f)[0]

        target_name = f"{base_name}_mask.png"
        dst_path = os.path.join(val_dir, target_name)

        try:
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            # WICHTIG: Auch Maske auf 512x512 zwingen!
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

            # Speichern als PNG
            cv2.imwrite(dst_path, img)
            count += 1
        except Exception as e:
            print(f"⚠️ Fehler bei Maske {f}: {e}")

        if i % 100 == 0:
            print(f"   ... {i}/{total} Masken verarbeitet", end='\r')

    print(f"\n✅ {count} Masken resized & konvertiert.")


def download_perceptual_loss():
    print("--- Check: Perceptual Loss Model ---")
    target_dir = "ade20k/ade20k-resnet50dilated-ppm_deepsup"
    target_file = os.path.join(target_dir, "encoder_epoch_20.pth")
    if os.path.exists(target_file):
        print("✅ Loss-Modell bereits vorhanden.")
        return
    print("⚠️ Loss-Modell fehlt. Lade herunter...")
    os.makedirs(target_dir, exist_ok=True)
    url = "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth"
    run_cmd(f"curl -L -o {target_file} {url}")
    print("✅ Download abgeschlossen.")


def prepare_data(bucket_name, debug_mode=False):
    print(f"--- 1. Daten Setup [Debug={debug_mode}] ---")
    start = time.time()

    if os.path.exists(LOCAL_DATA_ROOT): shutil.rmtree(LOCAL_DATA_ROOT)
    os.makedirs(LOCAL_TRAIN_DIR, exist_ok=True)
    os.makedirs(LOCAL_VAL_DIR, exist_ok=True)
    os.makedirs(LOCAL_REAL_VAL_DIR, exist_ok=True)

    # 1. TRAINING
    if debug_mode:
        folders_to_train = ["train_2"]
    else:
        folders_to_train = [f"train_{i}" for i in range(2, 10)]

    print(f"📥 Lade Training: {folders_to_train}")
    for folder in folders_to_train:
        src = f"gs://{bucket_name}/train/{folder}"
        try:
            run_cmd_silent(f"gcloud storage cp -r {src} {LOCAL_TRAIN_DIR}")
        except Exception as e:
            print(f"Warnung {folder}: {e}")
    flatten_directory(LOCAL_TRAIN_DIR)

    # 2. VALIDATION BILDER
    print(f"📥 Lade Validierung (Bilder)...")
    try:
        run_cmd_silent(f"gcloud storage cp -r gs://{bucket_name}/train/train_1 {LOCAL_VAL_DIR}")
    except Exception as e:
        print(f"⚠️ Fehler Val Bilder: {e}")
    flatten_directory(LOCAL_VAL_DIR)

    # NEU: Bilder auf 512x512 zwingen
    resize_images_in_dir(LOCAL_VAL_DIR, size=(512, 512))

    # 3. VALIDATION MASKEN
    print(f"📥 Lade Validierung (Masken)...")
    local_mask_temp = "/tmp/masks_temp"
    os.makedirs(local_mask_temp, exist_ok=True)
    try:
        run_cmd_silent(f"gcloud storage cp -r gs://{bucket_name}/train_1_mask {local_mask_temp}")
    except Exception as e:
        print(f"⚠️ Fehler Val Masken: {e}")
    flatten_directory(local_mask_temp)

    # 4. MERGE (mit Resize auf 512x512!)
    process_and_merge_masks(LOCAL_VAL_DIR, local_mask_temp)

    # 5. SYNC VAL
    src_files = os.listdir(LOCAL_VAL_DIR)
    for f in src_files:
        shutil.copy(os.path.join(LOCAL_VAL_DIR, f), LOCAL_REAL_VAL_DIR)

    # Final Check
    num_train = len([f for f in os.listdir(LOCAL_TRAIN_DIR) if f.endswith('.jpg')])
    num_val_img = len([f for f in os.listdir(LOCAL_REAL_VAL_DIR) if f.endswith('.jpg')])
    num_val_mask = len([f for f in os.listdir(LOCAL_REAL_VAL_DIR) if f.endswith('_mask.png')])

    print("\n" + "=" * 40)
    print(f"📊 REPORT:")
    print(f"   Trainings-Bilder: {num_train}")
    print(f"   Val-Bilder (512px): {num_val_img}")
    print(f"   Val-Masken (512px): {num_val_mask}")
    print("=" * 40 + "\n")

    if num_train == 0: raise RuntimeError("❌ Keine Trainingsdaten!")
    if num_val_img == 0: raise RuntimeError("❌ Keine Validierungsdaten!")
    if num_val_mask < num_val_img * 0.9:
        print("⚠️ WARNUNG: Masken fehlen! (Checke Dateinamen)")

    print(f"✅ Fertig in {(time.time() - start) / 60:.2f} min.")


def create_yaml_config():
    loc_path = "/app/lama/configs/training/location/my_cloud_data.yaml"
    with open(loc_path, "w") as f:
        f.write(
            f"data_root_dir: {LOCAL_DATA_ROOT}\n"
            f"out_root_dir: {LOCAL_MODEL_DIR}\n"
            f"tb_dir: {LOCAL_MODEL_DIR}/tb_logs"
        )

    # WICHTIG: KEIN img_suffix unter `train`, sonst knallt InpaintingTrainDataset
    data_content = """
defaults:
  - abl-04-256-mh-dist

# Train-Block weglassen oder nur Felder benutzen, die LaMa wirklich kennt.
# In deiner Version wird train sowieso über InpaintingTrainDataset(**, '*.jpg') gesteuert.

val:
  img_suffix: .jpg     # InpaintingDataset unterstützt img_suffix

visual_test:
  img_suffix: .jpg     # InpaintingEvalOnlineDataset unterstützt img_suffix
"""
    data_path = "/app/lama/configs/training/data/my_wikiart_data.yaml"
    with open(data_path, "w") as f:
        f.write(data_content)

    return "my_cloud_data", "my_wikiart_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    # Unbenutzte Args entfernt
    parser.add_argument('--data-folders', type=str, default="2-9")
    parser.add_argument('--fixed-masks', action='store_true')
    args = parser.parse_args()

    prepare_data(args.bucket, debug_mode=args.debug)
    download_perceptual_loss()

    loc_conf, data_conf = create_yaml_config()

    print(f"--- 2. Starte Big-Lama Training ---")

    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + "/lama"
    env['USER'] = "root"
    env['TORCH_HOME'] = os.getcwd()

    cmd = [
        sys.executable, "-u", "lama/bin/train.py",
        "-cn", "big-lama",
        f"location={loc_conf}",
        f"data={data_conf}",

        "data.batch_size=12",

        # WICHTIG: max_epochs im kwargs-Block des Trainers
        "trainer.kwargs.max_epochs={}".format(args.epochs),

        # Das hier kannst du lassen, wenn es vorher schon funktioniert hat:
        f"+trainer.resume_from_checkpoint={PRETRAINED_CKPT}",

        # !!! DIESE ZEILE RAUS:
        # "trainer.kwargs.precision=16",

        # log_every_n_steps ebenfalls im kwargs-Block
        "trainer.kwargs.log_every_n_steps=50",

        "optimizers.generator.lr=0.0001",

        # ENTSCHEIDEND: hier überschreiben wir das 25000-Ding:
        "trainer.kwargs.val_check_interval=1.0",  # 1.0 = einmal pro Epoche

        # HIER NEU:
        "trainer.kwargs.log_gpu_memory=null",     # GPU-Memory-Logging abschalten

        "hydra.run.dir=/tmp/experiments/hydra_logs",
    ]

    print(f"Startbefehl: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    for line in process.stdout:
        print(line, end='', flush=True)

    process.wait()

    if process.returncode == 0:
        print(f"--- 3. Upload nach gs://{args.bucket}/{OUTPUT_FOLDER_NAME} ---")
        if os.path.isdir(LOCAL_MODEL_DIR) and os.listdir(LOCAL_MODEL_DIR):
            try:
                run_cmd_silent(
                    f"gcloud storage cp -r {LOCAL_MODEL_DIR}/* gs://{args.bucket}/{OUTPUT_FOLDER_NAME}/"
                )
            except Exception as e:
                print(f"⚠️ Upload fehlgeschlagen: {e}")
        else:
            print("⚠️ Kein Inhalt in /tmp/experiments, überspringe Upload.")
    else:
        raise RuntimeError(f"Exit Code {process.returncode}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)