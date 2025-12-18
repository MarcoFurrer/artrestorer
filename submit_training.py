from google.cloud import aiplatform
import time
import subprocess

# --- KONFIGURATION ---
PROJECT_ID = "aicomp-477516"
REGION = "europe-west4"
BUCKET_NAME = "artrestorer"
JOB_NAME = "lama-wikiart-finetune-l4"  # Name angepasst

# DEBUG MODUS: False = Volles Training
DEBUG_MODE = False

# Docker Image
REPO_NAME = "vertex-ai-repo"
IMAGE_TAG = "lama-restorer-gpu:latest"
DOCKER_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{IMAGE_TAG}"


def build_and_push():
    print("--- Baue und pushe Docker Image (GPU Version) ---")
    subprocess.check_call([
        "docker", "build",
        "-f", "Dockerfile.finetuning",
        "--platform", "linux/amd64",
        "-t", DOCKER_IMAGE_URI,
        "."
    ])
    subprocess.check_call(["docker", "push", DOCKER_IMAGE_URI])


def submit_custom_job():
    print(f"--- Starte L4 Training auf Vertex AI ({REGION}) [Debug={DEBUG_MODE}] ---")

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_NAME,
        container_uri=DOCKER_IMAGE_URI,
        command=["python3", "training/task.py"],
        staging_bucket=f"gs://{BUCKET_NAME}"
    )

    training_args = [
        f"--bucket={BUCKET_NAME}",
        # ZEIT SPAREN: Nur 15 Epochen
        "--epochs=15"
    ]

    if DEBUG_MODE:
        training_args.append("--debug")

    print("Submitting Job... (Warte auf L4 GPU)")

    job.run(
        args=training_args,
        replica_count=1,
        # HARDWARE UPGRADE:
        machine_type="g2-standard-8",  # G2 ist für L4 optimiert
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        boot_disk_size_gb=300,
        sync=True
    )


if __name__ == "__main__":
    build_and_push()
    submit_custom_job()