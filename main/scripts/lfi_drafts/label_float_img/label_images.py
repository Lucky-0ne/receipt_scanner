import pandas as pd
import tkinter as tk
from tkinter import simpledialog
import cv2
from tqdm import tqdm
from PIL import ImageTk, Image
import numpy as np
import os

file_path_load = 'color_detection/images_float/all/'

root = tk.Tk()
root.title('Verify cropped image')
label = tk.Label(root)
label.pack()

# Optional: If you need to store additional information about images
image_data = []
for file_name_load in tqdm(os.listdir(file_path_load)):
    if file_name_load.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(file_path_load, file_name_load)
        img = cv2.imread(img_path)

        if img is not None:

            user_input = simpledialog.askstring("label float image", "put in the depcited float value:")

            if user_input is not None:
                # Store additional image information if needed
                image_info = {
                    'file_name': file_name_load,
                    'label': user_input,
                    # 'image_data': img  # You can store path instead of actual image data
                    'image_data': 'compressed image data'  # You can store path instead of actual image data
                }
            else:
                image_data.append(image_info)
                continue

        else:
            print(f"Failed to read image: {file_name_load}")
    else:
        print(f"Skipping non-image file: {file_name_load}")

# Creating a DataFrame
df = pd.DataFrame(image_data)