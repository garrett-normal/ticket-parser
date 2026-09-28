import cv2
import easyocr
import numpy as np
from img_functions import *
from matplotlib import pyplot as plt

reader = easyocr.Reader(['en'], gpu=True)
img_path = 'imgs/ticket1.jpg'
img = cv2.imread(img_path)

assert img is not None

corrected_skew = deskew(img)
grayscale_img = grayscale(corrected_skew)
inverted = invert_img(grayscale_img)
thicker = thicker_text(inverted)
write_img('generated/final.jpg',thicker)

result = reader.readtext('generated/final.jpg')

img = cv2.imread('generated/final.jpg')

draw_bounding_boxes(img,result)




# display_true_img_size(img_path)
# if img is not None:
#     plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGBA))
#     plt.show()