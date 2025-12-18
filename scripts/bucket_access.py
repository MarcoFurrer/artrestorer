import os
from google.cloud import storage

# Use environment variable for service account key path
key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if key_path:
    client = storage.Client.from_service_account_json(key_path)
else:
    client = storage.Client()

bucket = client.get_bucket('artrestorer')
# Jetzt haben sie Zugriff


import tensorflow as tf

# Das "**" findet alle Bilder rekursiv in allen Unterordnern
list_ds = tf.data.Dataset.list_files("gs://dein-bucket/train/*/*.jpg")

# Ergebnis: Du hast alle Bilder in einem Dataset, genau wie du wolltest.