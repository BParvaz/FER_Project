import glob
from PIL import Image, ImageDraw, ImageFont

pngs = sorted(glob.glob("logs/samples v2/epoch_*.png"))
frames = []

font = ImageFont.load_default()

for p in pngs:
    img = Image.open(p).convert("RGB")
    draw = ImageDraw.Draw(img)

    # extract epoch number from filename
    epoch = p.split("_")[-1].split(".")[0]
    text = f"Epoch {epoch}"

    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = img.width - w - 5
    y = img.height - h - 5

    # outline for contrast
    draw.text((x+1, y+1), text, fill="black", font=font)
    draw.text((x, y), text, fill="white", font=font)

    frames.append(img)

frames[0].save(
    "training v2.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)

print("Saved training.gif")