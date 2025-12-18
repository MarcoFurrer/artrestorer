from PIL import Image

# Load EPS
img = Image.open("destruct_images/overlays/zerrissene-zerrissene-papierblattkante-auf-transparentem-hintergrundvektor/14247.eps")

# Convert to RGBA (adds transparency)
img.load(scale=4, transparency=True)  # Higher scale = higher resolution
img = img.convert("RGBA")

# Save as transparent PNG
img.save("output.png", "PNG")