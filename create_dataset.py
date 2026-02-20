import pandas as pd
import numpy as np
import cv2
import os

df = pd.read_csv("sign_mnist_train.csv")

output_dir = "dataset"
images_per_class = 200  # 2 images per letter = ~52 images total

for label in df['label'].unique():
    folder = os.path.join(output_dir, str(label))
    os.makedirs(folder, exist_ok=True)
    
    subset = df[df['label'] == label].head(images_per_class)
    
    for i, (_, row) in enumerate(subset.iterrows()):
        pixels = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
        cv2.imwrite(os.path.join(folder, f"{i}.png"), pixels)

print("Dataset created!")