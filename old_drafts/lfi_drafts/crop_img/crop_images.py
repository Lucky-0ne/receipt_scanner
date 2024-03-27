import cv2
import numpy as np
import utils
import os
from tqdm import tqdm

def process_image(pathImage):
    img_orig = cv2.imread(pathImage)
    # widthImg_orig, heightImg_orig= Image.fromarray(img).size
    heightImg_orig, widthImg_orig, _ = img_orig.shape
    heightImg, widthImg = int(heightImg_orig/10), int(widthImg_orig/10)
    # heightImg = 640
    # widthImg  = 480
    img = cv2.resize(img_orig, (widthImg, heightImg)) # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    # thres=LO_utils.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    thres= (50, 200) # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    # kernel = np.ones((20, 20))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    imgThreshold_rescaled = cv2.resize(imgThreshold, (widthImg_orig, heightImg_orig))

    heightImg, widthImg = heightImg_orig, widthImg_orig

    ## FIND ALL COUNTOURS
    imgContours = img_orig.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img_orig.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, _ = cv2.findContours(imgThreshold_rescaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS

    # FIND THE BIGGEST COUNTOUR
    # TODO find better way
    biggest, _ = utils.biggestContour(contours) # FIND THE BIGGEST CONTOUR

    try:
        biggest=utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = utils.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img_orig, matrix, (widthImg, heightImg))

        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
    except:
        raise Exception("No contour found?")

    return imgWarpColored

image_path_load = 'images_raw/'
image_path_save = 'images_cropped/'
for file_name_load in tqdm(os.listdir(image_path_load)):
    processed_image = process_image(image_path_load + file_name_load)

    cv2.imwrite(image_path_save + file_name_load[:-4] + '_cropped.jpg', processed_image)