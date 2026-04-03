import cv2
import numpy as np
import os

os.makedirs("tags", exist_ok=True)

aruco = cv2.aruco
dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

for i in range(20):
    img = aruco.generateImageMarker(dict, i, 200)
    cv2.imwrite(f"tags/tag_{i}.png", img)
