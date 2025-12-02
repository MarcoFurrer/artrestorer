from PIL import Image
import numpy as np

# Load image
path = "destruct_images/overlays/fire.jpg"
img = Image.open(path)

# Convert to RGBA to support transparency
img = img.convert("RGBA")

# Convert to numpy array for efficient processing
data = np.array(img)

# Calculate luminance/intensity (brightness) of each pixel
# Using standard luminance formula: 0.299*R + 0.587*G + 0.114*B
luminance = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]

# Create alpha channel: bright pixels (>240) become transparent (alpha=0)
# Adjust threshold (240) if needed: higher = more white pixels made transparent
alpha = np.where(luminance > 240, 0, 255).astype(np.uint8)

# Set the alpha channel
data[:, :, 3] = alpha

# Convert back to PIL Image
result = Image.fromarray(data, 'RGBA')

# Save as transparent PNG
result.save("destruct_images/overlays/cracks_transparent.png", "PNG")
print("✓ Saved transparent PNG: destruct_images/overlays/cracks_transparent.png")