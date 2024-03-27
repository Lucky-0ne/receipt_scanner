from utils import *
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog
import os

plot = False

file_path_load = 'images_raw/'
file_path_save = 'images_float/'
# image_path = file_path_load + '2023-08-01_21-00.jpg'

# raw_image = cv2.imread(image_path)#[:,:,0]
# raw_image = cv2.resize(raw_image, (720, 1080))

# hsv_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

# if plot: plot_results(raw_image)

LOOKUP_TABLE = {
    "green": ["leon", np.array([30, 50, 20]), np.array([90, 255, 255])],
    "pink": ["tim", np.array([130, 50, 20]), np.array([170, 255, 255])],
    "orange": ["both", np.array([10, 50, 20]), np.array([30, 255, 255])]
}

GAP=20

# COLOR_TO_DETECT = "green"

root = tk.Tk()
root.title('Verify cropped image')
root.geometry("+350+0")
label = tk.Label(root)
label.pack()

for file_name_load in os.listdir(file_path_load):

    if file_name_load.endswith('.jpg'):
        image_path = file_path_load + file_name_load
        raw_image = cv2.imread(image_path)
        raw_image = cv2.resize(raw_image, (720, 1080))

        hsv_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

        if plot: plot_results(raw_image)
    else:
        raise ValueError('No image found')

    for COLOR_TO_DETECT in LOOKUP_TABLE.keys():
        owner, lower, upper = LOOKUP_TABLE.get(COLOR_TO_DETECT)

        # Create a binary mask for the specified color
        mask = cv2.inRange(hsv_image, lower, upper)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to store the regions inside bounding boxes
        result_image = raw_image.copy()

        boxs_coordinates = []
        # Draw bounding rectangles around the detected contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if all([w > 13, w < 40, h > 10, h < 44]):
                # print(w, h)
                x -= 90
                y -= 7
                w += 70
                h += 20
                cv2.rectangle(result_image, ((x), y), ((x) + w, y + h), (255, 0, 0), 3)
                boxs_coordinates.append((x, y, x + w, y + h))

            else:
                pass

        if plot: plot_results(result_image)

        # Create a black canvas with the same size as the original image
        canvas = np.zeros_like(result_image)

        image_snippets=[]
        # Copy the regions inside bounding boxes to the canvas
        for coordinates in boxs_coordinates:
            x1, y1, x2, y2 = coordinates
            canvas[y1:y2, x1:x2] = raw_image[y1:y2, x1:x2]
            image_snippets.append(raw_image[y1:y2, x1:x2])

        image_snippets=image_snippets[::-1]

        if plot: plot_results(canvas)

        if not image_snippets:
            print(f"No {COLOR_TO_DETECT} snippets found in {file_name_load}")
            continue

        # Define the dimensions of the canvas (you can adjust these as needed)
        max_width = np.max([Image.fromarray(te).width for te in image_snippets])
        total_height = np.sum([Image.fromarray(te).height for te in image_snippets])
        canvas_width = max_width
        canvas_height = total_height + GAP * len(image_snippets) - GAP

        # Create an empty canvas
        canvas_ordered = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

        x_offset = 0
        y_offset = 0
        for te in image_snippets:
            image = Image.fromarray(te)

            # Paste the image onto the canvas at the calculated position
            canvas_ordered.paste(image, (x_offset, y_offset))

            # Calculate the position to paste the image in the center of the canvas
            y_offset = (y_offset + image.height + GAP)

        canvas_array = np.array(canvas_ordered)

        # plot_results(canvas_array)
        # plot_results(image_snippets[0])
        # plot_results(image_snippets[-1])

        # photo = ImageTk.PhotoImage(image=Image.fromarray(canvas_array))
        check_image = np.hstack((cv2.resize(result_image, (480, 720)), cv2.resize(canvas, (480, 720))))
        photo = ImageTk.PhotoImage(image=Image.fromarray(check_image))
        label.configure(image=photo)
        label.image = photo  # Keep a reference to the image

        root.update()

        user_input = simpledialog.askstring('Proceed?', 'Press "Enter" to proceed, type "n" to skip')

        if user_input == 'n':
            print('Image snippets discarded')
            # for i, snippet in enumerate(image_snippets):
            #     cv2.imwrite(file_path_save + 'discarded/' + f'{file_name_load[:-4]}_{i}.jpg', snippet)
            continue
        else:
            for i, snippet in enumerate(image_snippets):
                cv2.imwrite(file_path_save + owner + f'/{file_name_load[:-4]}_{i}.jpg', snippet)
                cv2.imwrite(file_path_save + f'all/{file_name_load[:-4]}_{i}.jpg', snippet)
            print('Image snippets saved')

root.destroy()