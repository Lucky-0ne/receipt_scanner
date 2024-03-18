import cv2
import numpy as np
from PIL import Image
from IPython.display import display

def plot_results(image, mode="inline", scale=0.6):
    # Display the result
    if mode == "popup":
        pass
    elif mode == "inline":
        # Convert from BGR to RGB (because OpenCV uses BGR order for color channels, whereas PIL uses RGB.)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
    else:
        raise ValueError("Mode must be either 'popup' or 'inline'")

    if type(image) is np.ndarray:
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        width, height = image.size
        display(image.resize((int(width*scale), int(height*scale))))

COLOR_BOUNDS = {
    'green':{
        'lower': np.array([30, 50, 20]),
        'upper': np.array([90, 255, 255])
        },
    'orange':{
        'lower': np.array([10, 50, 20]),
        'upper': np.array([30, 255, 255])
        },
    'pink':{
        'lower': np.array([130, 50, 20]),
        'upper': np.array([170, 255, 255])
        }
}