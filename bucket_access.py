from google.cloud import storage

# Pfad zur JSON-Datei, die du ihnen geschickt hast
client = storage.Client.from_service_account_json('path/to/keyfile.json')
bucket = client.get_bucket('dein-bucket-name')
# Jetzt haben sie Zugriff