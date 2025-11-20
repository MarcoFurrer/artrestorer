from google.cloud import storage

# Pfad zur JSON-Datei, die du ihnen geschickt hast
client = storage.Client.from_service_account_json('aicomp-477516-a9f4b6a5785d.json')
bucket = client.get_bucket('artrestorer')
# Jetzt haben sie Zugriff