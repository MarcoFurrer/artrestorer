from PIL import Image
import os
import random
import numpy as np


def make_binary_mask(alpha_channel, threshold=127):
    """Convert alpha channel to binary mask (0 or 255)"""
    alpha_array = np.array(alpha_channel)
    binary = np.where(alpha_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def apply_crack(base_img, overlay_path, segment_count=3):
    """
    Overlay crack image, apply only random segments of crack image
    segment_count: divide image into this many segments and randomly pick one
    Returns: (result_image, mask) where mask is 1 where overlay exists, 0 elsewhere
    """
    if not os.path.exists(overlay_path):
        print(f"⚠ Crack overlay not found: {overlay_path}")
        # Return image and zero mask
        mask = Image.new("L", base_img.size, 0)
        return base_img, mask
    
    overlay = Image.open(overlay_path).convert("RGBA")
    
    # Divide overlay into segments and pick random one
    img_height = overlay.height
    segment_height = img_height // segment_count
    random_segment = random.randint(0, segment_count - 1)
    
    # Crop to random segment
    crop_box = (0, random_segment * segment_height, overlay.width, (random_segment + 1) * segment_height)
    overlay_segment = overlay.crop(crop_box)
    
    # Resize to base image width, keep aspect ratio
    scale_factor = base_img.width / overlay_segment.width
    new_height = int(overlay_segment.height * scale_factor)
    overlay_segment = overlay_segment.resize((base_img.width, new_height), Image.Resampling.LANCZOS)
    
    # Random vertical position
    max_y = max(0, base_img.height - overlay_segment.height)
    y_pos = random.randint(0, max_y) if max_y > 0 else 0
    
    # Create a transparent layer and paste the segment
    temp = base_img.copy()
    temp.paste(overlay_segment, (0, y_pos), overlay_segment)
    
    # Create mask: binary (0 or 255) where overlay is
    mask = Image.new("L", base_img.size, 0)
    overlay_alpha = overlay_segment.split()[3]
    # Convert to binary mask
    binary_alpha = make_binary_mask(overlay_alpha)
    mask.paste(binary_alpha, (0, y_pos))
    
    print(f"✓ Applied crack overlay (segment {random_segment})")
    return temp, mask


def apply_fire(base_img, overlay_path, scale_factor=0.3):
    """
    Overlay fire image, make it smaller and apply to random position
    scale_factor: size relative to base image (0.3 = 30% of base width)
    Returns: (result_image, mask) where mask is 1 where overlay exists, 0 elsewhere
    """
    if not os.path.exists(overlay_path):
        print(f"⚠ Fire overlay not found: {overlay_path}")
        mask = Image.new("L", base_img.size, 0)
        return base_img, mask
    
    overlay = Image.open(overlay_path).convert("RGBA")
    
    # Scale down fire
    new_width = int(base_img.width * scale_factor)
    scale = new_width / overlay.width
    new_height = int(overlay.height * scale)
    overlay = overlay.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Random position
    max_x = max(0, base_img.width - overlay.width)
    max_y = max(0, base_img.height - overlay.height)
    x_pos = random.randint(0, max_x) if max_x > 0 else 0
    y_pos = random.randint(0, max_y) if max_y > 0 else 0
    
    # Composite fire at random position
    temp = base_img.copy()
    temp.paste(overlay, (x_pos, y_pos), overlay)
    
    # Create mask: binary (0 or 255) where overlay is
    mask = Image.new("L", base_img.size, 0)
    overlay_alpha = overlay.split()[3]
    binary_alpha = make_binary_mask(overlay_alpha)
    mask.paste(binary_alpha, (x_pos, y_pos))
    
    print(f"✓ Applied fire overlay at position ({x_pos}, {y_pos})")
    return temp, mask


def apply_papercut(base_img, overlay_path, num_copies=5):
    """
    Overlay papercut image, move along x axis (distribute horizontally)
    num_copies: number of times to repeat across x axis
    Returns: (result_image, mask) where mask is 1 where overlay exists, 0 elsewhere
    """
    if not os.path.exists(overlay_path):
        print(f"⚠ Papercut overlay not found: {overlay_path}")
        mask = Image.new("L", base_img.size, 0)
        return base_img, mask
    
    overlay = Image.open(overlay_path).convert("RGBA")
    
    # Scale overlay to fit vertically on base
    scale = base_img.height / overlay.height
    new_width = int(overlay.width * scale)
    new_height = base_img.height
    overlay = overlay.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Distribute copies along x axis
    temp = base_img.copy()
    mask = Image.new("L", base_img.size, 0)
    x_spacing = base_img.width // (num_copies + 1)
    
    overlay_alpha = overlay.split()[3]
    binary_alpha = make_binary_mask(overlay_alpha)
    
    for i in range(num_copies):
        x_pos = x_spacing * (i + 1) - overlay.width // 2
        # Random vertical offset for variation
        y_offset = random.randint(-20, 20)
        y_pos = max(0, min(y_offset, base_img.height - overlay.height))
        
        temp.paste(overlay, (x_pos, y_pos), overlay)
        mask.paste(binary_alpha, (x_pos, y_pos))
    
    print(f"✓ Applied papercut overlay ({num_copies} copies along x axis)")
    return temp, mask




def main(base_image_path, output_path, mask_output_path, save_type="PNG"):
    # Load base image
    if not os.path.exists(base_image_path):
        print(f"Error: Base image not found: {base_image_path}")
        exit(1)

    base_img = Image.open(base_image_path).convert("RGBA")
    print(f"✓ Loaded base image: {base_img.size}")

    choice = random.randint(1, 3)
    print(f"✓ Randomly selected overlay type: {choice}")
    # Apply overlays using functions
    if choice == 1:
        base_img, mask = apply_crack(base_img, "destruct_images/overlays/transparent/cracks.png")
    elif choice == 2:
        base_img, mask = apply_fire(base_img, "destruct_images/overlays/transparent/fire.png", scale_factor=0.25)
    else:
        base_img, mask = apply_papercut(base_img, "destruct_images/overlays/transparent/papercut.png", num_copies=3)

    # Save result
    base_img.save(output_path, "PNG")
    mask.save(mask_output_path, "PNG")
    print(f"✓ Saved result: {output_path}")
    print(f"✓ Saved mask: {mask_output_path}")


if __name__ == "__main__":
    main("uploads/bild1.webp", "destruct_images/overlays/result_with_overlays.png", "destruct_images/overlays/mask.png")