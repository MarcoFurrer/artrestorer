from google.cloud import aiplatform
import time
import subprocess

# --- KONFIGURATION ---
PROJECT_ID = "aicomp-477516"
REGION = "europe-west4"

# ⚠️ ANGEPASST AUF DEIN BILD
BUCKET_NAME = "artrestorer"
JOB_NAME = "lama-wikiart-finetune-v1"

# Docker Image Name (wir geben ihm einen suffix -gpu zur Unterscheidung)
REPO_NAME = "vertex-ai-repo"
IMAGE_TAG = "lama-restorer-gpu:latest"
DOCKER_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{IMAGE_TAG}"


def build_and_push():
    print("--- Baue und pushe Docker Image (GPU Version) ---")

    # HIER IST DIE ÄNDERUNG: -f Dockerfile.finetuning
    subprocess.check_call([
        "docker", "build",
        "-f", "Dockerfile.finetuning",  # <--- Nimmt das spezielle File
        "--platform", "linux/amd64",
        "-t", DOCKER_IMAGE_URI,
        "."
    ])

    subprocess.check_call(["docker", "push", DOCKER_IMAGE_URI])


def submit_custom_job():
    print(f"--- Starte Training auf Vertex AI ({REGION}) ---")

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_NAME,
        container_uri=DOCKER_IMAGE_URI,
        command=["python3", "training/task.py"],
        staging_bucket=f"gs://{BUCKET_NAME}"
    )

    print("Submitting Job... (Warte auf GPU Zuweisung)")

    job.run(
        args=[
            f"--bucket={BUCKET_NAME}",
            "--epochs=20"
        ],
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        sync=True
    )


if __name__ == "__main__":
    build_and_push()
    submit_custom_job()