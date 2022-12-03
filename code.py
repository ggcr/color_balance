import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import os

def pull_image(url_path = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"):
    if not os.path.exists('lena.jpg'):
        im = requests.get(url_path).content
        with open('lena.jpg', 'wb') as handler:
            handler.write(im)
    return cv2.imread('lena.jpg')

def disp(cv2_im):
    color = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(color)

@interact
def sliders(RED=(0, 2, 0.1), GREEN=(0, 2, 0.1), BLUE=(0, 2, 0.1)):
    return color_balance(pull_image(), RED, GREEN, BLUE)

def color_balance(bgr_im, scal_R, scal_G, scal_B):
    for idx, scalar in list(zip([0,1,2], [scal_B, scal_G, scal_R])):
        table = [(i * scalar) for i in range(256)]
        table = np.array(table, np.uint8)
        bgr_im[:,:,idx] = cv2.LUT(bgr_im[:,:,idx], table)
    gammaCorrection(bgr_im, 2.2)
    disp(bgr_im)
    
def gammaCorrection(bgr_im, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(bgr_im, table)
