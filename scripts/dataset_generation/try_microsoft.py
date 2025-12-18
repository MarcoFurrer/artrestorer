#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
from destruct_images.microsoft_code import synthesize_speckle, synthesize_gaussian

PICTURE_PATH = "../uploads/bild1.webp"

pic_path = Path(__file__).parent / PICTURE_PATH

if not pic_path.exists():
    print(f"✗ Image not found at: {pic_path}")
else:
    print(f"Loading image from: {pic_path}")
    img = Image.open(pic_path)
    print(f"✓ Image loaded: {img.size} {img.mode}")
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Save original
    img.save(output_dir / "original.png")
    print(f"✓ Original saved")
    
    # Apply salt & pepper with different amounts
    for amount in [0.01, 0.05, 0.1]:
        print(f"Applying salt & pepper (amount={amount})...")
        noisy_img = synthesize_gaussian(img, std_l=5, std_r=50)
        output_path = output_dir / f"salt_pepper_{amount}.png"
        noisy_img.save(output_path)
        print(f"✓ Saved: {output_path.name}")
    
    print(f"\n✓ All results saved to: {output_dir}")
