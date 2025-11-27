import os
import subprocess
import time

# --- KONFIGURATION (Vom User übernommen & angepasst) ---
PROJECT_ID = "aicomp-477516"  # Deine Projekt-ID
REGION = "europe-west4"  # Deine Region

# Service Account (Bleibt gleich wie in deiner Vorlage)
SERVICE_ACCOUNT_EMAIL = "vertex-ai-runner@aicomp-477516.iam.gserviceaccount.com"

# Container- und Job-Namen
REPO_NAME = "vertex-ai-repo"  # Repository Name

# ⬇️ *** ANPASSUNGEN FÜR LAMA RESTORER *** ⬇️
NEW_APP_NAME = "lama-restorer-v1"
IMAGE_TAG = f"{NEW_APP_NAME}:latest"
CLOUD_RUN_SERVICE_NAME = "lama-restorer-api"

# WICHTIG: Port muss mit dem Dockerfile EXPOSE übereinstimmen (wir nutzen 8000)
CONTAINER_PORT = 8000
# ⬆️ ****************************************** ⬆️

DOCKER_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{IMAGE_TAG}"


# ----------------------------------------

def run_command(command, check=True, capture_output=False, **kwargs):
    """Führt einen Shell-Befehl aus und steuert die Ausgabe."""
    if isinstance(command, list):
        command = " ".join(command)

    print(f"\n---> Führe aus: {command}")

    result = subprocess.run(
        command,
        check=check,
        shell=True,
        capture_output=capture_output,
        text=True,
        **kwargs
    )
    if capture_output:
        return result
    return None


def build_and_push_docker_image():
    """Baut das Image, pusht es zur Artifact Registry und wartet kurz."""
    print("--- 1. Docker Image bauen ---")

    # 1. Artifact Registry Repository erstellen (falls nicht vorhanden)
    repo_create_cmd = f"gcloud artifacts repositories create {REPO_NAME} --repository-format=docker --location={REGION}"
    repo_result = run_command(repo_create_cmd, check=False, capture_output=True)

    if repo_result.returncode != 0 and "already exists" not in repo_result.stderr:
        print(f"\n❌ FEHLER beim Erstellen des Repositorys (Exit Code {repo_result.returncode}):")
        print("Details: Konnte das Artifact Registry Repository nicht erstellen.")
        # Wir machen trotzdem weiter, da es meistens daran liegt, dass es schon existiert

    print(f"Artifact Registry '{REPO_NAME}' ist verfügbar/geprüft.")

    # 2. Docker Image bauen (Nutzt Cache vom lokalen Docker, falls vorhanden)
    # --platform linux/amd64 ist wichtig für Cloud Run, falls du auf einem M1/M2 Mac bist
    build_command = ["docker", "build", "--platform", "linux/amd64", "-t", DOCKER_IMAGE_URI, "."]
    run_command(build_command, check=True)

    # 3. Docker Image pushen
    push_command = ["docker", "push", DOCKER_IMAGE_URI]
    print(f"\n--- 2. Docker Image pushen ({DOCKER_IMAGE_URI}) ---")
    run_command(push_command, check=True)

    print("\n✅ Image erfolgreich gebaut und gepusht.")
    print("Warte 5 Sekunden zur Synchronisierung...")
    time.sleep(5)


def deploy_cloud_run_service(image_uri: str, region: str):
    """
    Stellt das Image als öffentlichen Cloud Run Service bereit.
    """
    print(f"\n--- 3. Cloud Run Service bereitstellen ({CLOUD_RUN_SERVICE_NAME}) ---")
    print(f"Ziel-Port im Container: {CONTAINER_PORT}")

    deploy_cmd = [
        "gcloud", "run", "deploy", CLOUD_RUN_SERVICE_NAME,
        f"--image={image_uri}",
        f"--region={region}",
        f"--service-account={SERVICE_ACCOUNT_EMAIL}",

        # Port Konfiguration
        f"--port={CONTAINER_PORT}",

        # 🚨 RESSOURCEN ANPASSUNG FÜR LAMA 🚨
        # LaMa braucht RAM für das Modell (~1GB) + Bildpuffer.
        # 2Gi ist riskant (OOM Crash), 4Gi ist sicher.
        "--memory=4Gi",
        # 2 CPUs für schnellere Inferenz (da keine GPU)
        "--cpu=2",

        # Timeout erhöhen für große Bilder (Standard ist oft kurz)
        "--timeout=300",

        # Öffentlicher Zugriff
        "--allow-unauthenticated",
    ]

    run_command(deploy_cmd, check=True)

    print("\n✅ Service erfolgreich bereitgestellt.")


# ----------------------------------------
if __name__ == "__main__":

    # Kurzer Check ob Dateien da sind
    if not os.path.exists("big-lama"):
        print("❌ FEHLER: Ordner 'big-lama' fehlt! Bitte erst setup_lama.sh ausführen.")
        exit(1)

    try:
        # APIs aktivieren (Sicherheitshalber)
        run_command("gcloud services enable run.googleapis.com artifactregistry.googleapis.com", check=False)

        # Docker Auth
        print("--- Authentifizierung ---")
        auth_command = f"gcloud auth configure-docker {REGION}-docker.pkg.dev"
        run_command(auth_command, check=True)

        # Build & Push
        build_and_push_docker_image()

        # Deploy
        deploy_cloud_run_service(
            image_uri=DOCKER_IMAGE_URI,
            region=REGION
        )

    except subprocess.CalledProcessError as e:
        print(f"\n❌ FEHLER: Befehl fehlgeschlagen.")
    except Exception as e:
        print(f"\n❌ UNERWARTETER FEHLER: {e}")