from PIL import Image
import numpy as np

# Load image
path = "destruct_images/overlays/fire.png"
img = Image.open(path)

# Convert to RGBA to support transparency
img = img.convert("RGBA")

# Convert to numpy array for efficient processing
data = np.array(img)

# Calculate luminance/intensity (brightness) of each pixel
# Using standard luminance formula: 0.299*R + 0.587*G + 0.114*B
luminance = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]

# Also calculate color saturation to detect grey pixels (low saturation = grey/white)
# Saturation = (max - min) / max of RGB channels
r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
max_rgb = np.maximum(np.maximum(r, g), b)
min_rgb = np.minimum(np.minimum(r, g), b)
saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)

# Create alpha channel with gradient based on darkness:
# - Dark/black areas stay opaque (alpha=255)
# - Bright/white areas become transparent (alpha=0)
# - Inverse of luminance: darker = more opaque, brighter = more transparent

# Alpha inversely proportional to luminance
# At luminance 0 (black): alpha=255 (fully opaque)
# At luminance 255 (white): alpha=0 (fully transparent)
alpha = (255 - luminance).astype(np.uint8)

# Set the alpha channel
data[:, :, 3] = alpha

# Convert back to PIL Image
result = Image.fromarray(data, 'RGBA')

# Save as transparent PNG
result.save("destruct_images/overlays/fire_transparent.png", "PNG")
print("✓ Saved transparent PNG: destruct_images/overlays/fire_transparent.png")
