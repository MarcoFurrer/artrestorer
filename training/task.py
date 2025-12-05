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
LOCAL_REAL_VAL_DIR = os.path.join(LOCAL_DATA_ROOT, "val")
LOCAL_MODEL_DIR = "/tmp/experiments"
PRETRAINED_CKPT = "/app/big-lama/models/best.ckpt"


def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.check_call(cmd, shell=True)


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


def prepare_data(bucket_name, debug_mode=True):
    print(f"--- 1. Daten Download (High Speed) [Debug={debug_mode}] ---")
    start = time.time()

    os.makedirs(LOCAL_TRAIN_DIR, exist_ok=True)
    # Wir erstellen VAL Ordner nicht mit mkdir, da wir ihn gleich symlinken wollen

    # TRAINING DATA
    if debug_mode:
        print("⚡ DEBUG MODUS: Lade NUR train_2...")
        folders_to_train = ["train_2"]
    else:
        folders_to_train = [f"train_{i}" for i in range(2, 10)]

    print(f"Lade Ordner: {folders_to_train}")

    for folder in folders_to_train:
        src = f"gs://{bucket_name}/train/{folder}"
        try:
            target = os.path.join(LOCAL_TRAIN_DIR, folder)
            os.makedirs(target, exist_ok=True)
            run_cmd(f"gcloud storage cp -r {src}/* {target}")
        except Exception as e:
            print(f"⚠️ Warnung bei {folder}: {e}")

    # TEST DATA
    print(f"Lade Test-Daten nach {LOCAL_VAL_DIR}...")
    os.makedirs(LOCAL_VAL_DIR, exist_ok=True)
    try:
        if debug_mode: pass
        run_cmd(f"gcloud storage cp -r gs://{bucket_name}/test/test/* {LOCAL_VAL_DIR}")
    except Exception:
        try:
            run_cmd(f"gcloud storage cp -r gs://{bucket_name}/test/* {LOCAL_VAL_DIR}")
        except Exception as e:
            print(f"⚠️ Test-Daten Fehler: {e}")

    # VAL DATA (Optimierung: Symlink statt Kopie!)
    # Spart Speicherplatz und Zeit
    if os.path.exists(LOCAL_REAL_VAL_DIR):
        # Falls Ordner existiert (z.B. durch vorherigen Run), löschen
        if os.path.islink(LOCAL_REAL_VAL_DIR):
            os.unlink(LOCAL_REAL_VAL_DIR)
        else:
            shutil.rmtree(LOCAL_REAL_VAL_DIR)

    print(f"Erstelle Symlink für Validierung: {LOCAL_REAL_VAL_DIR} -> {LOCAL_VAL_DIR}")
    os.symlink(LOCAL_VAL_DIR, LOCAL_REAL_VAL_DIR)

    duration = (time.time() - start) / 60
    print(f"✅ Daten fertig in {duration:.2f} Minuten.")


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
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    prepare_data(args.bucket, debug_mode=args.debug)
    download_perceptual_loss()

    location_config = create_yaml_config()

    print("--- 2. Starte Big-Lama Training (Fine-Tuning) ---")

    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + "/lama"
    env['USER'] = "root"
    env['TORCH_HOME'] = os.getcwd()

    cmd = [
        sys.executable, "-u", "lama/bin/train.py",
        "-cn", "big-lama",
        f"location={location_config}",
        "data.batch_size=4",
        f"+trainer.max_epochs={args.epochs}",
        f"+trainer.resume_from_checkpoint={PRETRAINED_CKPT}",
        "+trainer.log_every_n_steps=50",
        "optimizers.generator.lr=0.0001",
        "hydra.run.dir=/tmp/experiments/hydra_logs",
        "data.train.img_suffix=.jpg",
        "data.val.img_suffix=.jpg",
        "data.visual_test.img_suffix=.jpg"
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
        run_cmd(f"gcloud storage cp -r {LOCAL_MODEL_DIR}/* gs://{args.bucket}/final_model/")
    except Exception as e:
        print(f"Upload Fehler: {e}")

    if process.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()