from PIL import Image
import os

# Base image to apply overlays to
base_image_path = "uploads/bild1.webp"

# List of transparent PNG overlays to apply
overlays = [
    # "destruct_images/overlays/transparent/fire.png",
    "destruct_images/overlays/transparent/cracks.png",
    # "destruct_images/overlays/transparent/papercut.png"

]

# Load base image
if not os.path.exists(base_image_path):
    print(f"Error: Base image not found: {base_image_path}")
    exit(1)

base_img = Image.open(base_image_path).convert("RGBA")
print(f"✓ Loaded base image: {base_img.size}")

# Apply each overlay
for overlay_path in overlays:
    if not os.path.exists(overlay_path):
        print(f"⚠ Overlay not found, skipping: {overlay_path}")
        continue
    
    # Load overlay
    overlay = Image.open(overlay_path).convert("RGBA")
    
    # Resize overlay to match base image if needed
    if overlay.size != base_img.size:
        overlay = overlay.resize(base_img.size, Image.Resampling.LANCZOS)
    
    # Composite: overlay on top of base using alpha channel
    base_img = Image.alpha_composite(base_img, overlay)
    print(f"✓ Applied overlay: {overlay_path}")

# Save result
output_path = "destruct_images/overlays/result_with_overlays.png"
base_img.save(output_path, "PNG")
print(f"✓ Saved result: {output_path}")
