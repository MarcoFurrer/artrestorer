import os
import sys
import subprocess
import argparse
import time
import shutil
import traceback

# --- PFADE ---
LOCAL_DATA_ROOT = "/tmp/dataset"
LOCAL_TRAIN_DIR = os.path.join(LOCAL_DATA_ROOT, "train")
LOCAL_VAL_DIR = os.path.join(LOCAL_DATA_ROOT, "visual_test")
LOCAL_REAL_VAL_DIR = os.path.join(LOCAL_DATA_ROOT, "val")
LOCAL_MODEL_DIR = "/tmp/experiments"
PRETRAINED_CKPT = "/app/big-lama/models/best.ckpt"


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
    print(f"🔨 Flattening directory: {directory} ...")
    count = 0
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            source = os.path.join(root, file)
            target = os.path.join(directory, file)
            if root != directory:
                try:
                    shutil.move(source, target)
                    count += 1
                except Exception as e:
                    print(f"Fehler beim Verschieben von {file}: {e}")
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except:
                pass
    print(f"✅ {count} Dateien flachgeklopft.")


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


def prepare_data(bucket_name, data_folders_arg, debug_mode=False):
    print(f"--- 1. Daten Download & Strukturierung [Debug={debug_mode}] ---")
    start = time.time()

    if os.path.exists(LOCAL_DATA_ROOT):
        shutil.rmtree(LOCAL_DATA_ROOT)

    os.makedirs(LOCAL_TRAIN_DIR, exist_ok=True)
    os.makedirs(LOCAL_VAL_DIR, exist_ok=True)
    os.makedirs(LOCAL_REAL_VAL_DIR, exist_ok=True)

    # TRAINING
    folders_to_train = []
    if debug_mode:
        print("⚡ DEBUG MODUS: Lade nur train_2...")
        folders_to_train = ["train_2"]
    elif "-" in data_folders_arg:
        start_idx, end_idx = map(int, data_folders_arg.split("-"))
        folders_to_train = [f"train_{i}" for i in range(start_idx, end_idx + 1)]
    else:
        folders_to_train = [f"train_{data_folders_arg}"]

    print(f"Lade Ordner: {folders_to_train}")
    for folder in folders_to_train:
        src = f"gs://{bucket_name}/train/{folder}"
        try:
            run_cmd_silent(f"gcloud storage cp -r {src} {LOCAL_TRAIN_DIR}")
        except Exception as e:
            print(f"⚠️ Warnung bei {folder}: {e}")
    flatten_directory(LOCAL_TRAIN_DIR)

    # TEST/VAL
    print(f"Lade Test-Daten...")
    try:
        run_cmd_silent(f"gcloud storage cp -r gs://{bucket_name}/test/test {LOCAL_VAL_DIR}")
    except Exception:
        try:
            run_cmd_silent(f"gcloud storage cp -r gs://{bucket_name}/test {LOCAL_VAL_DIR}")
        except Exception as e:
            print(f"⚠️ Test-Daten Fehler: {e}")
    flatten_directory(LOCAL_VAL_DIR)

    # COPY TO VAL
    print(f"Kopiere Daten von {LOCAL_VAL_DIR} nach {LOCAL_REAL_VAL_DIR}...")
    src_files = os.listdir(LOCAL_VAL_DIR)
    for file_name in src_files:
        full_file_name = os.path.join(LOCAL_VAL_DIR, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, LOCAL_REAL_VAL_DIR)

    # CHECK
    num_train_files = len(
        [name for name in os.listdir(LOCAL_TRAIN_DIR) if os.path.isfile(os.path.join(LOCAL_TRAIN_DIR, name))])
    num_val_files = len(
        [name for name in os.listdir(LOCAL_REAL_VAL_DIR) if os.path.isfile(os.path.join(LOCAL_REAL_VAL_DIR, name))])

    print("\n" + "=" * 40)
    print(f"📊 STATUS REPORT:")
    print(f"   Trainings-Bilder: {num_train_files}")
    print(f"   Validierungs-Bilder: {num_val_files}")
    print("=" * 40 + "\n")

    if num_train_files == 0: raise RuntimeError("❌ FEHLER: Keine Trainings-Bilder!")
    if num_val_files == 0: raise RuntimeError("❌ FEHLER: Keine Validierungs-Bilder!")

    print(f"✅ Daten fertig in {(time.time() - start) / 60:.2f} Minuten.")


def create_yaml_config(use_fixed_masks=False):
    # 1. Location
    loc_path = "/app/lama/configs/training/location/my_cloud_data.yaml"
    with open(loc_path, "w") as f:
        f.write(f"data_root_dir: {LOCAL_DATA_ROOT}\nout_root_dir: {LOCAL_MODEL_DIR}\ntb_dir: {LOCAL_MODEL_DIR}/tb_logs")

    # 2. Data Config (Hardcoded Fix für Phase 1)
    if use_fixed_masks:
        # Phase 2 (Später)
        print("🎭 CONFIG: Fixed Masks")
        mask_block = "mask_generator: null"
        suffix_line = "mask_suffix: _mask.png"
    else:
        # Phase 1 (JETZT): Wir definieren den Generator explizit für VAL!
        print("🎲 CONFIG: Random Masks (Hardcoded for Val)")
        mask_block = """
  mask_generator:
    kind: irregular
    kwargs:
      max_angle: 4
      max_len: 200
      max_width: 100
      max_times: 5
      min_times: 1
        """
        suffix_line = ""

    # Wir nutzen hier mask_block auch für TRAIN, um sicher zu sein,
    # aber wichtig ist er für VAL und VISUAL_TEST.
    data_content = f"""
defaults:
  - abl-04-256-mh-dist

train:
  img_suffix: .jpg
  {suffix_line}

val:
  img_suffix: .jpg
  {mask_block}
  {suffix_line}

visual_test:
  img_suffix: .jpg
  {mask_block}
  {suffix_line}
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
    parser.add_argument('--data-folders', type=str, default="2-9")
    parser.add_argument('--fixed-masks', action='store_true')
    args = parser.parse_args()

    prepare_data(args.bucket, args.data_folders, debug_mode=args.debug)
    download_perceptual_loss()

    loc_conf, data_conf = create_yaml_config(use_fixed_masks=args.fixed_masks)

    print(f"--- 2. Starte Big-Lama Training [FixedMasks={args.fixed_masks}] ---")

    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + "/lama"
    env['USER'] = "root"
    env['TORCH_HOME'] = os.getcwd()

    # WICHTIG: Die ++ Overrides für img_suffix sind weg, da wir sie jetzt sauber
    # im Yaml-File (data_conf) definiert haben. Das ist viel stabiler!
    cmd = [
        sys.executable, "-u", "lama/bin/train.py",
        "-cn", "big-lama",
        f"location={loc_conf}",
        f"data={data_conf}",
        "data.batch_size=4",
        f"+trainer.max_epochs={args.epochs}",
        f"+trainer.resume_from_checkpoint={PRETRAINED_CKPT}",
        "+trainer.log_every_n_steps=50",
        "optimizers.generator.lr=0.0001",
        "hydra.run.dir=/tmp/experiments/hydra_logs"
    ]

    print(f"Startbefehl: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    for line in process.stdout:
        print(line, end='', flush=True)

    process.wait()

    print(f"--- 3. Upload Ergebnisse ---")
    try:
        run_cmd_silent(f"gcloud storage cp -r {LOCAL_MODEL_DIR}/* gs://{args.bucket}/final_model/")
    except Exception as e:
        print(f"Upload Fehler: {e}")

    if process.returncode != 0:
        raise RuntimeError(f"Exit Code {process.returncode}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)