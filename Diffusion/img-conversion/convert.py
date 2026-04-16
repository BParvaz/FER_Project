import os
import pandas as pd
import numpy as np
from PIL import Image

# path to FER2013 csv
csv_path = "../../data/FER2013/train.csv"
test_path = "../../data/FER2013/test.csv"

# output folder
output_dir = "fer_images"

# emotion labels
emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# load dataset
train_df = pd.read_csv(csv_path)
test_df = pd.read_csv(csv_path)

for i, row in train_df.iterrows():

    emotion = emotion_map[row["emotion"]]

    # create folder if needed
    os.makedirs(f"{output_dir}/train", exist_ok=True)

    # convert pixel string to array
    pixels = np.array(row["pixels"].split(), dtype=np.uint8)

    # reshape to image
    img = pixels.reshape(48, 48)

    # convert to PIL image
    image = Image.fromarray(img)

    # save
    image.save(f"{output_dir}/train/{emotion}_{i}.png")

print("done with train")

for i, row in test_df.iterrows():

    emotion = emotion_map[row["emotion"]]

    # create folder if needed
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    # convert pixel string to array
    pixels = np.array(row["pixels"].split(), dtype=np.uint8)

    # reshape to image
    img = pixels.reshape(48, 48)

    # convert to PIL image
    image = Image.fromarray(img)

    # save
    image.save(f"{output_dir}/test/{emotion}_{i}.png")

print("done with test")