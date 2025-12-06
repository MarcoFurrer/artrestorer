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

# Source Ordner (Rohdaten für das Tool)
LOCAL_VAL_SOURCE = os.path.join(LOCAL_DATA_ROOT, "val_source")
LOCAL_TEST_SOURCE = os.path.join(LOCAL_DATA_ROOT, "visual_test_source")

# Ziel Ordner (Fertig generierte PNGs mit Masken)
LOCAL_VAL_TARGET = os.path.join(LOCAL_DATA_ROOT, "val")
LOCAL_TEST_TARGET = os.path.join(LOCAL_DATA_ROOT, "visual_test")

LOCAL_MODEL_DIR = "/tmp/experiments"
PRETRAINED_CKPT = "/app/big-lama/models/best.ckpt"


def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.check_call(cmd, shell=True)


def run_cmd_silent(cmd):
    print(f"Executing (silent): {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"⚠️ FEHLER bei: {cmd}")
        print("🔁 Wiederhole laut...")
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
                except:
                    pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except:
                pass
    print(f"✅ {count} Dateien flachgeklopft.")


def generate_official_masks(input_dir, output_dir, config_name="random_medium_512.yaml"):
    """
    Ruft das offizielle LaMa Tool 'gen_mask_dataset.py' auf.
    Wandelt JPGs in PNGs um und erstellt Masken.
    """
    print(f"🏭 Starte LaMa Mask Generator: {input_dir} -> {output_dir}")

    # Pfad zur Config im Container
    config_path = f"/app/lama/configs/data_gen/{config_name}"

    if not os.path.exists(config_path):
        # Fallback falls 512 nicht existiert (manche Repos haben nur 256)
        print(f"⚠️ Config {config_name} nicht gefunden, versuche random_medium_256.yaml")
        config_path = "/app/lama/configs/data_gen/random_medium_256.yaml"

    # Das offizielle Script aufrufen
    # Syntax: python bin/gen_mask_dataset.py <config> <indir> <outdir> --ext jpg
    cmd = f"python3 /app/lama/bin/gen_mask_dataset.py {config_path} {input_dir} {output_dir} --ext jpg"

    # Wir müssen PYTHONPATH setzen, damit das Skript seine Module findet
    env = os.environ.copy()
    env['PYTHONPATH'] = "/app/lama"

    subprocess.check_call(cmd, shell=True, env=env)
    print("✅ Masken Generierung abgeschlossen.")


def download_perceptual_loss():
    target_dir = "ade20k/ade20k-resnet50dilated-ppm_deepsup"
    target_file = os.path.join(target_dir, "encoder_epoch_20.pth")
    if os.path.exists(target_file): return
    os.makedirs(target_dir, exist_ok=True)
    run_cmd(
        f"curl -L -o {target_file} http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth")


def prepare_data(bucket_name, data_folders_arg, debug_mode=False):
    print(f"--- 1. Daten Download & Processing ---")
    start = time.time()

    if os.path.exists(LOCAL_DATA_ROOT): shutil.rmtree(LOCAL_DATA_ROOT)

    # Ordner erstellen
    os.makedirs(LOCAL_TRAIN_DIR, exist_ok=True)
    os.makedirs(LOCAL_VAL_SOURCE, exist_ok=True)  # Source für Generator
    # Target Ordner werden vom Script erstellt oder müssen leer sein

    # 1. TRAINING (Bleibt JPG, da on-the-fly masking)
    folders_to_train = []
    if debug_mode:
        folders_to_train = ["train_2"]
    elif "-" in data_folders_arg:
        s, e = map(int, data_folders_arg.split("-"))
        folders_to_train = [f"train_{i}" for i in range(s, e + 1)]
    else:
        folders_to_train = [f"train_{data_folders_arg}"]

    print(f"Lade Training: {folders_to_train}")
    for folder in folders_to_train:
        try:
            run_cmd_silent(f"gcloud storage cp -r gs://{bucket_name}/train/{folder} {LOCAL_TRAIN_DIR}")
        except:
            pass
    flatten_directory(LOCAL_TRAIN_DIR)

    # 2. VAL & TEST (Laden in SOURCE Ordner)
    print(f"Lade Test-Daten (Source)...")
    try:
        # Wir laden die Daten erst in val_source
        run_cmd_silent(f"gcloud storage cp -r gs://{bucket_name}/test/test {LOCAL_VAL_SOURCE}")
    except:
        run_cmd_silent(f"gcloud storage cp -r gs://{bucket_name}/test {LOCAL_VAL_SOURCE}")
    flatten_directory(LOCAL_VAL_SOURCE)

    # 3. GENERATOR LAUFEN LASSEN
    # Erzeugt saubere Daten in LOCAL_VAL_TARGET (val)
    # Wir nutzen dieselben Daten für Visual Test
    generate_official_masks(LOCAL_VAL_SOURCE, LOCAL_VAL_TARGET)
    generate_official_masks(LOCAL_VAL_SOURCE, LOCAL_TEST_TARGET)

    # Final Check
    num_train = len(os.listdir(LOCAL_TRAIN_DIR))
    num_val = len(os.listdir(LOCAL_VAL_TARGET))
    print(f"📊 Train JPGs: {num_train} | Val PNGs (Masked): {num_val}")

    if num_train == 0 or num_val == 0: raise RuntimeError("Daten fehlen!")
    print(f"✅ Fertig in {(time.time() - start) / 60:.2f} min.")


def create_yaml_config():
    loc_path = "/app/lama/configs/training/location/my_cloud_data.yaml"
    with open(loc_path, "w") as f:
        f.write(f"data_root_dir: {LOCAL_DATA_ROOT}\nout_root_dir: {LOCAL_MODEL_DIR}\ntb_dir: {LOCAL_MODEL_DIR}/tb_logs")

    # WICHTIG: Val & Test sind jetzt .png (vom Generator erstellt)
    # Train bleibt .jpg (on-the-fly)
    data_content = """
defaults:
  - abl-04-256-mh-dist

train:
  img_suffix: .jpg

val:
  img_suffix: .png
  # Keine Masken-Generierung mehr nötig, da Files existieren!

visual_test:
  img_suffix: .png
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

    print(f"--- 3. Upload ---")
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
        traceback.print_exc(); sys.exit(1)