import os
import sys
import subprocess
import argparse
import time
import shutil

# --- PFADE ---
LOCAL_DATA_ROOT = "/tmp/dataset"
LOCAL_TRAIN_DIR = os.path.join(LOCAL_DATA_ROOT, "train")
LOCAL_VAL_DIR = os.path.join(LOCAL_DATA_ROOT, "visual_test")
LOCAL_MODEL_DIR = "/tmp/experiments"
# Pfad zum mitgelieferten Modell im Docker Container
PRETRAINED_CKPT = "/app/big-lama/models/best.ckpt"


def run_gsutil(cmd):
    # -q für quiet (weniger Log-Spam beim Download), -m für Multithreading
    full_cmd = f"gsutil -m -q {cmd}"
    print(f"Executing: {full_cmd}")
    subprocess.check_call(full_cmd, shell=True)


def prepare_data(bucket_name):
    print("--- 1. Daten Download (Selektiv: train_2 bis train_9) ---")
    start = time.time()

    os.makedirs(LOCAL_TRAIN_DIR, exist_ok=True)
    os.makedirs(LOCAL_VAL_DIR, exist_ok=True)

    # 1. Training Data: NUR train_2 bis train_9 laden
    folders_to_train = [f"train_{i}" for i in range(2, 10)]

    print(f"Lade folgende Ordner: {folders_to_train}")

    for folder in folders_to_train:
        src = f"gs://{bucket_name}/train/{folder}"
        try:
            target = os.path.join(LOCAL_TRAIN_DIR, folder)
            os.makedirs(target, exist_ok=True)
            # cp -r kopiert den Inhalt
            run_gsutil(f"cp -r {src}/* {target}")
        except Exception as e:
            print(f"⚠️ Warnung bei {folder}: {e}")

    # 2. Test Data (Validation)
    try:
        run_gsutil(f"cp -r gs://{bucket_name}/test/* {LOCAL_VAL_DIR}")
    except Exception as e:
        print(f"Warnung bei Test-Data: {e}")

    duration = (time.time() - start) / 60
    print(f"✅ Download fertig in {duration:.2f} Minuten.")


def create_yaml_config():
    """Konfiguriert Pfade"""
    config_content = f"""
        data_root_dir: {LOCAL_DATA_ROOT}
        out_root_dir: {LOCAL_MODEL_DIR}
        tb_dir: {LOCAL_MODEL_DIR}/tb_logs
    """
    config_path = "/app/lama/configs/training/location/my_cloud_data.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)
    return "my_cloud_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    prepare_data(args.bucket)
    location_config = create_yaml_config()

    print("--- 2. Starte LaMa Training (Fine-Tuning) ---")

    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + "/lama"

    # Argumente zusammenbauen
    cmd = [
        sys.executable, "-u", "lama/bin/train.py",
        "-cn", "lama-fourier",
        f"location={location_config}",
        "data.batch_size=8",

        # --- FIX HIER: Das Pluszeichen (+) ist entscheidend! ---

        # Fügt max_epochs zur Config hinzu
        f"+trainer.max_epochs={args.epochs}",

        # Fügt Checkpoint-Pfad hinzu
        f"+trainer.resume_from_checkpoint={PRETRAINED_CKPT}",

        # Fügt Log-Frequenz hinzu
        "+trainer.log_every_n_steps=50",

        # Learning Rate (Existiert meistens schon, aber + schadet hier nicht falls deep nested)
        "optimizers.generator.optimizer_params.lr=0.0001"
    ]

    print(f"Startbefehl: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream Output
    for line in process.stdout:
        print(line, end='', flush=True)

    process.wait()

    print(f"--- 3. Upload Ergebnisse nach gs://{args.bucket}/final_model ---")
    try:
        # Wir laden hoch, auch wenn Training failt (für Logs im tb_logs Ordner)
        run_gsutil(f"cp -r {LOCAL_MODEL_DIR}/* gs://{args.bucket}/final_model/")
    except Exception as e:
        print(f"Upload Fehler: {e}")

    if process.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()