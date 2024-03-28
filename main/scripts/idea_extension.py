from credentials import stream_url
from utils import COLOR_BOUNDS
import cv2
import numpy as np
from datetime import datetime

# original frame size
ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 1920, 1080

# rescaled frame size
FRAME_HEIGHT = 960
FRAME_WIDTH = 540

# scale factors for ROI box
ROI_WIDTH = 350
ROI_HEIGHT = 20

# frame size of "pure" color box to unify shapes
C_BOX_WIDTH, C_BOX_HEIGHT = 10, 10

# Calculate scaling factors
width_scale = ORIGINAL_WIDTH / FRAME_WIDTH
height_scale = ORIGINAL_HEIGHT / FRAME_HEIGHT

# scroll bar stuff for detection frame
cv2.namedWindow('Frame')
tl_hor = int(FRAME_WIDTH * 0)
tl_ver = int(FRAME_HEIGHT * 0)
br_hor = int(FRAME_WIDTH * 1)
br_ver = int(FRAME_HEIGHT * 1)
def update_tl_hor(val):
    global tl_hor
    tl_hor = int(FRAME_WIDTH * (val/100))
def update_tl_ver(val):
    global tl_ver
    tl_ver = int(FRAME_HEIGHT * (val/100))
def update_br_hor(val):
    global br_hor
    br_hor = int(FRAME_WIDTH * (val/100))
def update_br_ver(val):
    global br_ver
    br_ver = int(FRAME_HEIGHT * (val/100))

# Create trackbars for the box size
cv2.createTrackbar('TL Horizontal', 'Frame', 0, 100, update_tl_hor)
cv2.createTrackbar('TL Vertical', 'Frame', 0, 100, update_tl_ver)
cv2.createTrackbar('BR Horizontal', 'Frame', 100, 100, update_br_hor)
cv2.createTrackbar('BR Vertical', 'Frame', 100, 100, update_br_ver)

# scroll bar stuff for the box offset
offset_x = 0
offset_y = 0
offset_width = 0
offset_height = 0
def update_offset_x(val):
    global offset_x
    offset_x = val
def update_offset_y(val):
    global offset_y
    offset_y = val
def update_offset_width(val):
    global offset_width
    offset_width = val
def update_offset_height(val):
    global offset_height
    offset_height = val
cv2.createTrackbar('x offset', 'Frame', 300, 350, update_offset_x)
cv2.createTrackbar('y offset', 'Frame', 5, 20, update_offset_y)
cv2.createTrackbar('width offset', 'Frame', 290, 350, update_offset_width)
cv2.createTrackbar('height offset', 'Frame', 10, 20, update_offset_height)

# Create a VideoCapture object (life feed from IP app on phone)
cap = cv2.VideoCapture(stream_url)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

idx_c=0 # index for color bounds
color_name = list(COLOR_BOUNDS.keys())[idx_c]
c_bound = COLOR_BOUNDS[color_name]
lower, upper = c_bound['lower'], c_bound['upper'] # initialize color bounds
show_box, zoom = True, False

while True:
    key = cv2.waitKey(1) & 0xFF
    ret, frame_orig = cap.read()
    if not ret:
        print("Cannot receive frame (stream end?). Exiting ...")
        break
    
    # Resize the displayed frame
    frame = cv2.resize(frame_orig, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

    # toggle box display and zoom
    if key == ord('b'):
        show_box = not show_box
    if key == ord('z'):
        zoom = not zoom
    
    if show_box:

        # Draw a red rectangle on the frame if show_box is True
        frame_top_left = (tl_hor, tl_ver)
        frame_bottom_right = (br_hor, br_ver)
        cv2.rectangle(frame, frame_top_left, frame_bottom_right, (0, 0, 255), 3)  # Draw rectangle

        # Crop the frame to the specified box
        cropped_frame = frame[tl_ver:br_ver, tl_hor:br_hor]
        cropped_frame_copy = cropped_frame.copy()

        hsv_img = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        # Create a binary mask for the specified color
        mask = cv2.inRange(hsv_img, lower, upper)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_coordinates = []
        # Draw bounding rectangles around the detected contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if all([w >= 5, w <= 15, h >= 5, h <= 15]):
                w, h = C_BOX_WIDTH, C_BOX_HEIGHT    # Set the width and height within threshold to a fixed size to unify output shapes
                # apply slider offset
                x -= offset_x
                y -= offset_y
                w += offset_width
                h += offset_height
                # draw boxes
                cv2.rectangle(cropped_frame, ((x), y), ((x) + w, y + h), (255, 0, 0), 3)
                # collect all box coordinates
                # box_coordinates.append((x, y, x + w, y + h))
                box_coordinates.append((x, y, w, h))
        box_coordinates = box_coordinates[::-1]     # Reverse the box coordinates to match the order of the boxes in the canvas

    if zoom and box_coordinates:  # Ensure there's at least one box to zoom into

        # Calculate required canvas size
        num_boxes = len(box_coordinates)
        # canvas_cols = int(np.ceil(np.sqrt(num_boxes)))
        # canvas_rows = int(np.ceil(num_boxes / canvas_cols))
        canvas_cols = 1
        canvas_rows = num_boxes
        canvas_width = canvas_cols * ROI_WIDTH
        canvas_height = canvas_rows * ROI_HEIGHT
        
        # Create a blank canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Process and display each ROI
        for idx_box, (x, y, w, h) in enumerate(box_coordinates):
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:  # Check if coordinates are within frame bounds
                roi = cropped_frame_copy[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (ROI_WIDTH, ROI_HEIGHT))

                row = idx_box // canvas_cols
                col = idx_box % canvas_cols
                canvas[row*ROI_HEIGHT:(row+1)*ROI_HEIGHT, col*ROI_WIDTH:(col+1)*ROI_WIDTH] = roi_resized
            cv2.imshow(f'ROIs', canvas)

    # Display the resulting frame
    cv2.imshow('IP Webcam', frame)
    
    # Break the loop
    if key == ord('q'):
        break
    # switch color (detection) mode
    if key == ord('c'):
        idx_c+=1
        if idx_c == len(COLOR_BOUNDS): idx_c = 0    # reset index to close the cycle
        color_name = list(COLOR_BOUNDS.keys())[idx_c]
        c_bound = COLOR_BOUNDS[color_name]
        lower, upper = c_bound['lower'], c_bound['upper']
    
    # save ROIs as .png image snippets with timestamp
    if key == ord('s'):
            print(f"Saving {len(box_coordinates)} ROIs as images...")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # rescale displayed snippets to original resolution
            for idx_s, (x, y, w, h) in enumerate(box_coordinates):

                # Adjust box coordinates back to original frame's scale
                orig_x, orig_y = int(x * width_scale), int(y * height_scale)
                orig_w, orig_h = int(w * width_scale), int(h * height_scale)
                
                orig_tl_ver, orig_br_ver = int(tl_ver*height_scale), int(br_ver*height_scale)
                orig_tl_hor, orig_br_hor = int(tl_hor*width_scale), int(br_hor*width_scale)

                # Extract the ROI box
                cropped_frame_orig = frame_orig[orig_tl_ver:orig_br_ver, orig_tl_hor:orig_br_hor]
                roi = cropped_frame_orig[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]

                # Define a file path to save the image (e.g., in the current directory with an incremental index)
                file_path = f"{timestamp}_{color_name}_ROI_{idx_s}.png"

                # Save the ROI image
                cv2.imwrite(f"images/result_snippets/{color_name}/"+file_path, roi)
                print(f"Saved {file_path}")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()