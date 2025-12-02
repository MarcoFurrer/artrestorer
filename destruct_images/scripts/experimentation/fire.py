from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import random
import numpy as np

IMAGE_PATH = "uploads/bild1.webp"
OUTPUT_PATH = "uploads/bild1_burnt.webp"
INTENSITY = 0.5  # 0-1

img = Image.open(IMAGE_PATH).convert("RGB")
width, height = img.size

# Increase contrast for more dramatic effect
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(1 + INTENSITY * 0.5)

# Darken significantly
enhancer = ImageEnhance.Brightness(img)
img = enhancer.enhance(1 - INTENSITY * 0.6)

# Desaturate
enhancer = ImageEnhance.Color(img)
img = enhancer.enhance(1 - INTENSITY * 0.5)

# Convert to numpy for better pixel manipulation
img_array = np.array(img, dtype=np.float32)

# Apply heavy color shift towards brown/orange/black
img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + INTENSITY * 0.6) + INTENSITY * 60, 0, 255)  # Red boost
img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 - INTENSITY * 0.5), 0, 255)  # Green reduce
img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - INTENSITY * 0.7), 0, 255)  # Blue reduce heavily

# Add burnt edges with gradient burn effect
burn_edge_width = int(min(width, height) * INTENSITY * 0.25)
for i in range(burn_edge_width):
    ratio = i / burn_edge_width
    burn_intensity = (1 - ratio) * INTENSITY
    
    # Create burnt frame
    img_array[:i, :] = img_array[:i, :] * (1 - burn_intensity * 0.8)
    img_array[-i-1:, :] = img_array[-i-1:, :] * (1 - burn_intensity * 0.8)
    img_array[:, :i] = img_array[:, :i] * (1 - burn_intensity * 0.8)
    img_array[:, -i-1:] = img_array[:, -i-1:] * (1 - burn_intensity * 0.8)

# Add large charred areas with irregular shapes (not perfect circles)
num_large_burns = int(INTENSITY * 6)
for _ in range(num_large_burns):
    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)
    base_radius = random.randint(25, 100)
    
    # Create irregular burnt area using perlin-like randomness
    for y in range(max(0, cy - base_radius), min(height, cy + base_radius)):
        for x in range(max(0, cx - base_radius), min(width, cx + base_radius)):
            dist = ((x - cx)**2 + (y - cy)**2) ** 0.5
            # Add randomness to create irregular edges
            random_factor = 0.7 + random.random() * 0.3
            effective_radius = base_radius * random_factor
            
            if dist <= effective_radius:
                darkness = 1 - (dist / effective_radius) * 0.4
                img_array[y, x] *= darkness * (0.2 + random.random() * 0.2)

# Add smaller burn spots with varied intensity
num_small_burns = int(INTENSITY * 20)
for _ in range(num_small_burns):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    radius = random.randint(3, 20)
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= nx < width and 0 <= ny < height:
                dist = (dx**2 + dy**2) ** 0.5
                if dist <= radius:
                    # Irregular burn with random intensity
                    burn_strength = 0.3 + random.random() * 0.4
                    img_array[ny, nx] *= burn_strength

# Add subtle texture/noise instead of heavy blur
noise = np.random.normal(0, 2, img_array.shape)
img_array = np.clip(img_array + noise * INTENSITY, 0, 255)

# Minimal blur - just to blend edges, not make it blurry
img_array = Image.fromarray(np.uint8(np.clip(img_array, 0, 255)))
img_array = img_array.filter(ImageFilter.GaussianBlur(radius=0.5))

img_array.save(OUTPUT_PATH)
print(f"✓ Burnt image saved to: {OUTPUT_PATH}")