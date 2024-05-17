'''

File name: test_image_filtering.py 
Description: This script performs filtering image

Author: Victor Santiago Solis Garcia
Creation date: 05/16/2024

Usage example:  python .\stereo-vision.py --l_img .\left_image.png --r_img .\right_image.png

'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Definir constantes fuera de las funciones
BASELINE = 94.926
RECTIFIED_FX = 648.52
RECTIFIED_CX = 635.709
RECTIFIED_CY = 370.88

def parser_user_data():
    """
    Function to input the user data
    
    Parameter(s):    None

    Returns:      args object
    """
    parser = argparse.ArgumentParser(description="Stereo images")
    parser.add_argument('--l_img', type=str, required=True, help="Path to the left image.")
    parser.add_argument('--r_img', type=str, required=True, help="Path to the right image.")
    args = parser.parse_args()
    return args

def compute_values(uL, vL, uR, vR, baseline=BASELINE, fx=RECTIFIED_FX, cx=RECTIFIED_CX, cy=RECTIFIED_CY):
    """
    Function to compute values 
    
    Parameter(s):    uL,vl,Ur,vR,baseline,fx,cx,cy

    Returns:      X,Y,Z coordinates
    """
    Ucl = uL - cx
    Vcl = vL - cy
    Ucr = uR - cx
    d = Ucl - Ucr
    Z = (fx * baseline) / d
    X = Ucl * (Z / fx)
    Y = Vcl * (Z / fx)
    return X, Y, Z

def select_points_left(event, x, y, flags, param):
    """
    Function to map 3d points, save points selected and compute disparity
    
    Parameter(s):    event,x,y

    Returns:      None
    """
    points_left, image_left, disparity_map = param
    if event == cv2.EVENT_LBUTTONDOWN:
        points_left.append((x, y))
        cv2.circle(image_left, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Select Points - Left Image', image_left)
        if len(points_left) % 30 == 0:  
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i, point in enumerate(points_left):
                uL, vL = point
                disparity = round(disparity_map[vL, uL])
                uR = round(uL - disparity)
                X, Y, Z = compute_values(uL, vL, uR, vL)
                if i == 0:  
                    ax.scatter(X, Y, Z, c='r', marker='o')
                    ax.text(X, Y, Z, "Primer Punto", color='black')
                elif i == len(points_left) - 1: 
                    ax.scatter(X, Y, Z, c='g', marker='o')
                    ax.text(X, Y, Z, "Último Punto", color='black')
                else:
                    ax.scatter(X, Y, Z, c='b', marker='o')  
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

def pipeline():
    args = parser_user_data()

    image_left = cv2.imread(args.l_img, 0)  
    image_right = cv2.imread(args.r_img, 0)

    stereo = cv2.StereoBM_create(numDisparities=144, blockSize=23)

    # Calcular mapa de disparidad
    disparity_map = stereo.compute(image_left, image_right).astype(np.float32) / 16

    points_left = []

    cv2.namedWindow('Select Points - Left Image')
    cv2.setMouseCallback('Select Points - Left Image', select_points_left, [points_left, image_left, disparity_map])
    cv2.imshow('Select Points - Left Image', image_left)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pipeline()
