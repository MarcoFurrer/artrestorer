from PIL import Image
import numpy as np
import random
import math

IMAGE_PATH = "uploads/bild1.webp"
OUTPUT_PATH = "uploads/bild1_teared.webp"
TEAR_INTENSITY = 0.7  # 0-1

def process_image(image_path, output_path, tear_intensity=0.7):
    """Apply tear effect with holes and separated displaced parts."""
    
    # Open image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Create background (white/light gray for visible holes)
    result = np.full((height, width, 3), [220, 220, 220], dtype=np.float32)
    img_array = np.array(img, dtype=np.float32)
    
    # Copy original image as base
    result[:, :, :] = img_array.copy()
    
    # Step 1: Create large irregular holes
    num_holes = int(tear_intensity * 8)
    for _ in range(num_holes):
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)
        
        # Irregular hole shape
        hole_radius = random.randint(int(width * 0.04), int(width * 0.15))
        
        for y in range(max(0, center_y - hole_radius), min(height, center_y + hole_radius)):
            for x in range(max(0, center_x - hole_radius), min(width, center_x + hole_radius)):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Create irregular jagged edge
                roughness = random.randint(-5, 5)
                
                if dist < hole_radius + roughness:
                    # Gradually fade to background towards edges
                    fade = max(0, (hole_radius + roughness - dist) / 10)
                    result[y, x, :] = result[y, x, :] * (1 - fade * 0.3) + np.array([220, 220, 220]) * fade * 0.3
                
                if dist < hole_radius - 5:
                    # Make hole - show background
                    result[y, x, :] = [220, 220, 220]
    
    # Step 2: Create torn/separated fragments - displace parts of image
    num_fragments = int(tear_intensity * 6)
    for _ in range(num_fragments):
        # Random fragment area
        frag_x = random.randint(0, width - 1)
        frag_y = random.randint(0, height - 1)
        frag_w = random.randint(int(width * 0.08), int(width * 0.25))
        frag_h = random.randint(int(height * 0.08), int(height * 0.25))
        
        # Random displacement
        disp_x = random.randint(-80, 80)
        disp_y = random.randint(-80, 80)
        
        # Copy fragment to new location with rotation/skew effect
        for y in range(frag_h):
            for x in range(frag_w):
                src_y = frag_y + y
                src_x = frag_x + x
                
                dst_y = frag_y + y + disp_y
                dst_x = frag_x + x + disp_x
                
                if (0 <= src_y < height and 0 <= src_x < width and 
                    0 <= dst_y < height and 0 <= dst_x < width):
                    
                    # Get original pixel
                    pixel = img_array[src_y, src_x, :].copy()
                    
                    # Darken for shadow effect
                    pixel *= 0.8
                    
                    # Blend with existing (semi-transparent overlay)
                    result[dst_y, dst_x, :] = result[dst_y, dst_x, :] * 0.3 + pixel * 0.7
    
    # Step 3: Create jagged torn lines that split sections
    num_tears = int(tear_intensity * 5)
    for tear_idx in range(num_tears):
        # Starting point
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)
        
        # Direction
        angle = random.uniform(0, 2 * math.pi)
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Tear length
        tear_length = random.randint(int(height * 0.3), int(height * 0.7))
        
        curr_x = float(start_x)
        curr_y = float(start_y)
        
        # Draw torn line
        for step in range(tear_length):
            curr_x += dx + random.uniform(-0.8, 0.8)
            curr_y += dy + random.uniform(-0.8, 0.8)
            
            if not (0 <= int(curr_y) < height and 0 <= int(curr_x) < width):
                break
            
            # Create jagged gap with varying width
            tear_width = random.randint(3, 12)
            
            for offset in range(-tear_width, tear_width + 1):
                for offset2 in range(-tear_width // 2, tear_width // 2 + 1):
                    px = int(curr_x) + offset
                    py = int(curr_y) + offset2
                    
                    if 0 <= px < width and 0 <= py < height:
                        dist = math.sqrt(offset**2 + offset2**2)
                        
                        if dist < tear_width * 0.6:
                            # Make gap - show background
                            result[py, px, :] = [220, 220, 220]
                        else:
                            # Edge - darken
                            fade = max(0, 1 - dist / tear_width)
                            result[py, px, :] *= (1 - fade * 0.6)
    
    # Step 4: Add small scattered torn pieces
    num_scattered = int(tear_intensity * 12)
    for _ in range(num_scattered):
        # Small random rectangular pieces
        piece_x = random.randint(0, width - 1)
        piece_y = random.randint(0, height - 1)
        piece_w = random.randint(3, 25)
        piece_h = random.randint(3, 25)
        
        # Displacement
        disp_x = random.randint(-150, 150)
        disp_y = random.randint(-150, 150)
        
        # Copy piece
        for y in range(piece_h):
            for x in range(piece_w):
                src_y = piece_y + y
                src_x = piece_x + x
                dst_y = piece_y + y + disp_y
                dst_x = piece_x + x + disp_x
                
                if (0 <= src_y < height and 0 <= src_x < width and
                    0 <= dst_y < height and 0 <= dst_x < width):
                    
                    pixel = img_array[src_y, src_x, :].copy()
                    pixel *= random.uniform(0.5, 0.9)  # Darken
                    
                    # Add shadow/edge
                    result[dst_y, dst_x, :] = pixel
    
    # Step 5: Add damage texture and edges
    for y in range(height):
        for x in range(width):
            if random.random() < tear_intensity * 0.1:
                # Random noise/texture on torn areas
                noise = random.randint(-30, 30)
                result[y, x, :] = np.clip(result[y, x, :] + noise, 0, 255)
    
    # Convert and save
    result_img = Image.fromarray(np.uint8(np.clip(result, 0, 255)))
    result_img.save(output_path)
    print(f"Teared image with holes and fragments saved to {output_path}")

if __name__ == "__main__":
    process_image(IMAGE_PATH, OUTPUT_PATH, TEAR_INTENSITY)
