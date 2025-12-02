from PIL import Image, ImageFilter
import numpy as np
import random
import math

IMAGE_PATH = "uploads/bild1.webp"
OUTPUT_PATH = "uploads/bild1_washed.webp"
WASH_LEVEL = 0.5       # How much of image is affected by water (0-1)
STRENGTH = 0.7         # Intensity of wash effect (0-1)
ROUGHNESS = 0.6        # Edge irregularity (0-1)
WATER_COLOR = "#A0B0C0"  # Light blue-gray water stain

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def process_image(image_path, output_path, wash_level=0.5, strength=0.7, roughness=0.6, water_color="#A0B0C0"):
    """Apply water-damaged effect to image."""
    
    # Open image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Parse water color
    water_rgb = hex_to_rgb(water_color)
    
    # Clamp parameters
    wash_level = max(0, min(1, wash_level))
    strength = max(0, min(1, strength))
    roughness = max(0, min(1, roughness))
    
    # Apply color desaturation for faded look
    for y in range(height):
        for x in range(width):
            r = img_array[y, x, 0]
            g = img_array[y, x, 1]
            b = img_array[y, x, 2]
            
            # Convert to grayscale
            gray = (r + g + b) / 3
            
            # Blend original color with grayscale for faded effect
            fade_factor = strength * wash_level * 0.3
            img_array[y, x, 0] = r * (1 - fade_factor) + gray * fade_factor
            img_array[y, x, 1] = g * (1 - fade_factor) + gray * fade_factor
            img_array[y, x, 2] = b * (1 - fade_factor) + gray * fade_factor
    
    # Increase brightness (water-washed look)
    img_array = np.clip(img_array * (1 + strength * 0.2), 0, 255)
    
    # Create water damage pattern from bottom up
    center_y = height * (1 - wash_level * 0.3)  # Water damage starts from bottom
    center_x = width / 2
    max_dist = math.sqrt((width/2)**2 + (height/2)**2)
    
    # Add vertical water streaks
    num_streaks = int(wash_level * 8)
    for _ in range(num_streaks):
        streak_x = random.randint(0, width - 1)
        streak_start_y = int(height * (1 - wash_level))
        
        for y in range(streak_start_y, height):
            # Streak gets narrower as it goes down
            streak_width = int((1 - (y - streak_start_y) / (height - streak_start_y)) * random.randint(2, 8))
            
            for x in range(max(0, streak_x - streak_width), min(width, streak_x + streak_width)):
                if 0 <= y < height and 0 <= x < width:
                    # Blend with water color
                    blend = 0.4 * strength
                    img_array[y, x, 0] = img_array[y, x, 0] * (1 - blend) + water_rgb[0] * blend
                    img_array[y, x, 1] = img_array[y, x, 1] * (1 - blend) + water_rgb[1] * blend
                    img_array[y, x, 2] = img_array[y, x, 2] * (1 - blend) + water_rgb[2] * blend
    
    # Apply water wash effect from bottom (radial from bottom-center)
    for y in range(height):
        for x in range(width):
            # Distance from bottom-center
            dx = x - center_x
            dy = y - center_y
            dist_from_center = math.sqrt(dx**2 + dy**2)
            
            # Normalized distance
            if max_dist > 0:
                norm_dist = dist_from_center / max_dist
            else:
                norm_dist = 0
            
            # Add roughness (water edge irregularity)
            roughness_magnitude = roughness * 0.15
            roughness_effect = (random.random() - 0.5) * 2.0 * roughness_magnitude
            norm_dist = max(0, min(1, norm_dist + roughness_effect))
            
            # Calculate wash factor
            pixel_wash_factor = 0
            if wash_level > 1e-6:
                wash_start_radius = 1.0 - wash_level
                
                if norm_dist >= wash_start_radius:
                    pixel_wash_factor = (norm_dist - wash_start_radius) / wash_level
                    pixel_wash_factor = max(0, min(1, pixel_wash_factor))
                    pixel_wash_factor = pixel_wash_factor ** 1.5
            
            # Apply water damage effect
            if pixel_wash_factor > 0:
                effect_strength = pixel_wash_factor * strength
                
                # Blend with water color and desaturate
                gray = (img_array[y, x, 0] + img_array[y, x, 1] + img_array[y, x, 2]) / 3
                
                # Increase lightness
                r = img_array[y, x, 0] * (1 - effect_strength) + (gray + water_rgb[0]) / 2 * effect_strength
                g = img_array[y, x, 1] * (1 - effect_strength) + (gray + water_rgb[1]) / 2 * effect_strength
                b = img_array[y, x, 2] * (1 - effect_strength) + (gray + water_rgb[2]) / 2 * effect_strength
                
                img_array[y, x, 0] = max(0, min(255, r))
                img_array[y, x, 1] = max(0, min(255, g))
                img_array[y, x, 2] = max(0, min(255, b))
    
    # Convert to image for filtering
    result_img = Image.fromarray(np.uint8(img_array))
    
    # Add slight blur for water-damaged effect
    result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    # Save result
    result_img.save(output_path)
    print(f"✓ Water-damaged image saved to: {output_path}")

# Run the effect
process_image(IMAGE_PATH, OUTPUT_PATH,
              wash_level=WASH_LEVEL,
              strength=STRENGTH,
              roughness=ROUGHNESS,
              water_color=WATER_COLOR)
