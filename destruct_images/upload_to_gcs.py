from google.cloud import storage
import os
from pathlib import Path

# Initialize GCS client
client = storage.Client.from_service_account_json('aicomp-477516-a9f4b6a5785d.json')
bucket = client.get_bucket('artrestorer')

# Define directories to upload
base_path = Path('')
destructed_dir = base_path / 'test_destructed'
mask_dir = base_path / 'test_mask'

def upload_directory(local_dir, gcs_prefix):
    """Upload all files from a local directory to GCS with a given prefix"""
    if not local_dir.exists():
        print(f"Directory not found: {local_dir}")
        return
    
    files = list(local_dir.glob('*.jpg'))
    print(f"Found {len(files)} files in {local_dir}")
    
    for i, file_path in enumerate(files, 1):
        # Create GCS blob path
        gcs_path = f"{gcs_prefix}/{file_path.name}"
        
        # Upload file
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(file_path))
        
        if i % 100 == 0:
            print(f"  Uploaded {i}/{len(files)} files...")
    
    print(f"Finished uploading {len(files)} files from {gcs_prefix}")

#upload original images
print("Uploading destructed images...")
upload_directory(destructed_dir, 'test_marco')

# Upload destructed images
print("Uploading destructed images...")
upload_directory(destructed_dir, 'test_destructed')

# Upload mask images
print("\nUploading mask images...")
upload_directory(mask_dir, 'test_mask')
print("\nAll uploads complete!")