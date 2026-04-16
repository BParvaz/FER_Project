import numpy as np

from PIL import Image
import os


data = np.load("./../img_samples/samples_64x64x64x3.npz")
print(data.files)
images = data["arr_0"]
print(images.shape)
os.makedirs("outputs", exist_ok=True)

for i, img in enumerate(images):
    Image.fromarray(img).save(f"outputs/sample_{i}.png")