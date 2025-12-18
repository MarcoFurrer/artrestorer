from overlay_transparent import main
import os
from multiprocessing import Pool, cpu_count

def process_file(file):
    """Process single file"""
    try:
        main(
            os.path.join("test", file),
            os.path.join("test_destructed", file),
            os.path.join("test_mask", file),
            save_type="JPEG",
        )
        print(f"✓ {file}")
    except Exception as e:
        print(f"✗ {file}: {e}")

if __name__ == "__main__":
    files = [f for f in os.listdir("test")]
    
    # Use all available CPU cores
    num_workers = cpu_count()
    print(f"Processing {len(files)} files with {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        pool.map(process_file, files)
    
    print("Done!")