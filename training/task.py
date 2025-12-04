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
PRETRAINED_CKPT = "/app/big-lama/models/best.ckpt"


def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.check_call(cmd, shell=True)


def download_perceptual_loss():
    """Lädt das fehlende ResNet-Modell für die Loss-Berechnung"""
    print("--- Check: Perceptual Loss Model ---")
    # Zielpfad relativ zu TORCH_HOME (was wir auf . setzen)
    target_dir = "ade20k/ade20k-resnet50dilated-ppm_deepsup"
    target_file = os.path.join(target_dir, "encoder_epoch_20.pth")

    if os.path.exists(target_file):
        print("✅ Loss-Modell bereits vorhanden.")
        return

    print("⚠️ Loss-Modell fehlt. Lade herunter...")
    os.makedirs(target_dir, exist_ok=True)
    url = "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth"
    # Download mit curl
    run_cmd(f"curl -L -o {target_file} {url}")
    print("✅ Download abgeschlossen.")


def prepare_data(bucket_name):
    print("--- 1. Daten Download (Selektiv & Flach) ---")
    start = time.time()

    os.makedirs(LOCAL_TRAIN_DIR, exist_ok=True)
    os.makedirs(LOCAL_VAL_DIR, exist_ok=True)

    # 1. Training Data (train_2 bis train_9)
    folders_to_train = [f"train_{i}" for i in range(2, 10)]

    for folder in folders_to_train:
        src = f"gs://{bucket_name}/train/{folder}"
        try:
            target = os.path.join(LOCAL_TRAIN_DIR, folder)
            os.makedirs(target, exist_ok=True)
            # -m für Multithreading, -q für Ruhe
            run_cmd(f"gsutil -m -q cp -r {src}/* {target}")
        except Exception as e:
            print(f"⚠️ Warnung bei {folder}: {e}")

    # 2. Test Data
    print(f"Lade Test-Daten...")
    try:
        run_cmd(f"gsutil -m -q cp -r gs://{bucket_name}/test/test/* {LOCAL_VAL_DIR}")
    except Exception:
        run_cmd(f"gsutil -m -q cp -r gs://{bucket_name}/test/* {LOCAL_VAL_DIR}")

    duration = (time.time() - start) / 60
    print(f"✅ Daten Download fertig in {duration:.2f} Minuten.")


def create_yaml_config():
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

    # 1. Downloads
    prepare_data(args.bucket)
    download_perceptual_loss()  # NEU: Verhindert den nächsten Crash

    location_config = create_yaml_config()

    print("--- 2. Starte Big-Lama Training (Fine-Tuning) ---")

    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + "/lama"
    env['USER'] = "root"
    # FIX: Setze TORCH_HOME auf das aktuelle Verzeichnis (/app),
    # damit er den Ordner 'ade20k' findet, den wir oben erstellt haben.
    env['TORCH_HOME'] = os.getcwd()

    cmd = [
        sys.executable, "-u", "lama/bin/train.py",
        "-cn", "big-lama",
        f"location={location_config}",
        "data.batch_size=4",
        f"+trainer.max_epochs={args.epochs}",
        f"+trainer.resume_from_checkpoint={PRETRAINED_CKPT}",
        "+trainer.log_every_n_steps=50",
        "+optimizers.generator.optimizer_params.lr=0.0001",
        "hydra.run.dir=/tmp/experiments/hydra_logs"
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

    for line in process.stdout:
        print(line, end='', flush=True)

    process.wait()

    print(f"--- 3. Upload Ergebnisse nach gs://{args.bucket}/final_model ---")
    try:
        run_cmd(f"gsutil -m -q cp -r {LOCAL_MODEL_DIR}/* gs://{args.bucket}/final_model/")
    except Exception as e:
        print(f"Upload Fehler: {e}")

    if process.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()