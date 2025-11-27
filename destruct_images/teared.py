from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random
import math

IMAGE_PATH = "uploads/bild1.webp"
OUTPUT_PATH = "uploads/bild1_teared.webp"
TEAR_INTENSITY = 0.7  # 0-1, intensity of tear effect
NUM_TEARS = 15         # Number of tear lines
FOLD_INTENSITY = 0.8   # How visible folds/creases are

def process_image(image_path, output_path, tear_intensity=0.7, num_tears=15, fold_intensity=0.8):
    """Apply severe tear/rip/folding effect to image like destroyed paper."""
    
    # Open image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Convert to numpy array for pixel manipulation
    img_array = np.array(img, dtype=np.float32)
    destroyed_array = img_array.copy()
    
    # Step 1: Add large missing/torn sections with irregular edges
    num_destroyed_sections = int(tear_intensity * 6)
    for _ in range(num_destroyed_sections):
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)
        
        section_width = random.randint(int(width * 0.08), int(width * 0.3))
        section_height = random.randint(int(height * 0.08), int(height * 0.4))
        
        end_x = min(width, start_x + section_width)
        end_y = min(height, start_y + section_height)
        
        # Create irregular destroyed section with jagged edges
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                # Use Perlin-like noise for jagged edges
                distance_to_edge = min(
                    x - start_x, 
                    y - start_y, 
                    end_x - x, 
                    end_y - y
                )
                edge_roughness = random.randint(-15, 15)
                
                if distance_to_edge + edge_roughness < 5:
                    # Rough edge - gradually fade out
                    alpha = max(0, (5 - distance_to_edge - edge_roughness) / 10)
                    destroyed_array[y, x, :] = destroyed_array[y, x, :] * (1 - alpha * 0.8)
                elif random.random() > 0.15:
                    # Inside destroyed area - make it darker/missing
                    destroyed_array[y, x, :] *= 0.1  # Very dark
    
    # Step 2: Add major displaced/folded sections - shift large areas of the image
    num_folds = int(fold_intensity * 4)
    for fold_num in range(num_folds):
        # Random rectangular section to displace
        section_x = random.randint(0, int(width * 0.7))
        section_y = random.randint(0, int(height * 0.7))
        section_w = random.randint(int(width * 0.15), int(width * 0.4))
        section_h = random.randint(int(height * 0.15), int(height * 0.4))
        
        # Displacement offset (simulating folding/movement)
        offset_x = random.randint(-80, 80)
        offset_y = random.randint(-80, 80)
        
        # Copy and displace this section
        for y in range(max(0, section_y), min(height, section_y + section_h)):
            for x in range(max(0, section_x), min(width, section_x + section_w)):
                new_x = x + offset_x
                new_y = y + offset_y
                
                if 0 <= new_x < width and 0 <= new_y < height:
                    # Blend the displaced section with darkening
                    src_pixel = img_array[y, x, :]
                    dest_pixel = destroyed_array[new_y, new_x, :]
                    
                    # Overlay with slight transparency and darkening for folded effect
                    blend_factor = 0.6
                    destroyed_array[new_y, new_x, :] = (
                        dest_pixel * (1 - blend_factor) + 
                        src_pixel * blend_factor * 0.7  # Darken the displaced part
                    )
        
        # Add a visible crease/fold line at edges
        crease_width = random.randint(3, 8)
        if random.random() > 0.5:
            # Vertical crease at section edge
            crease_x = section_x + random.randint(5, section_w - 5)
            for y in range(max(0, section_y), min(height, section_y + section_h)):
                for cx in range(max(0, crease_x - crease_width // 2), min(width, crease_x + crease_width // 2)):
                    destroyed_array[y, cx, :] *= 0.3  # Very dark crease
    
    # Step 3: Generate major jagged tear lines with wide destruction
    for tear_num in range(num_tears):
        start_x = random.randint(int(width * 0.05), int(width * 0.95))
        start_y = random.randint(int(height * 0.05), int(height * 0.95))
        
        # Tear can go in any direction
        angle = random.uniform(0, 2 * math.pi)
        tear_dx = math.cos(angle) * 0.8
        tear_dy = math.sin(angle) * 0.8
        
        tear_width = max(8, int(tear_intensity * 35))
        tear_length = random.randint(int(height * 0.3), int(height * 0.8))
        
        current_x = float(start_x)
        current_y = float(start_y)
        
        for step in range(tear_length):
            # Add heavy irregularity to the tear path
            jitter_x = random.uniform(-1.5, 1.5)
            jitter_y = random.uniform(-1.5, 1.5)
            current_x += tear_dx + jitter_x
            current_y += tear_dy + jitter_y
            
            if not (0 <= int(current_y) < height and 0 <= int(current_x) < width):
                break
            
            # Draw very thick tear with strong destruction
            for offset in range(-tear_width // 2, tear_width // 2 + 1):
                for offset2 in range(-tear_width // 3, tear_width // 3 + 1):
                    tear_x = int(current_x) + offset
                    tear_y = int(current_y) + offset2
                    
                    if 0 <= tear_x < width and 0 <= tear_y < height:
                        distance = math.sqrt(offset**2 + offset2**2)
                        fade = max(0, 1 - (distance / (tear_width / 2)))
                        
                        # Destroy the tear area significantly
                        destroy_factor = fade * tear_intensity
                        
                        # Darken heavily
                        destroyed_array[tear_y, tear_x, :] *= (1 - destroy_factor * 0.95)
                        
                        # Add some color shift for torn appearance
                        destroyed_array[tear_y, tear_x, :] += np.array([5, 5, 5]) * destroy_factor * 20
    
    # Step 4: Add extreme crumpling texture and damage staining
    for y in range(height):
        for x in range(width):
            damage_chance = tear_intensity * 0.35
            
            if random.random() < damage_chance:
                # Add heavy texture/burn variation
                noise_intensity = random.randint(-80, 80)
                for c in range(3):
                    destroyed_array[y, x, c] = np.clip(destroyed_array[y, x, c] + noise_intensity, 0, 255)
            
            # Add staining/discoloration for paper damage
            if random.random() < tear_intensity * 0.2:
                # Brown/dark staining from moisture/burning
                stain_intensity = random.uniform(0.4, 0.8)
                stain_color = np.array([random.randint(60, 120), random.randint(50, 100), random.randint(40, 80)])
                destroyed_array[y, x, :] = (
                    destroyed_array[y, x, :] * (1 - stain_intensity) + 
                    stain_color * stain_intensity
                )
    
    # Convert back to image
    result_img = Image.fromarray(np.uint8(np.clip(destroyed_array, 0, 255)))
    result_img.save(output_path)
    print(f"Destroyed/teared image saved to {output_path}")

if __name__ == "__main__":
    process_image(IMAGE_PATH, OUTPUT_PATH, TEAR_INTENSITY, NUM_TEARS, FOLD_INTENSITY)
