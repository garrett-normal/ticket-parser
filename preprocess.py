import cv2
import easyocr
import numpy as np
from img_functions import *
from matplotlib import pyplot as plt

reader = easyocr.Reader(['en'])
img_path = 'imgs/ticket1.jpg'
img = cv2.imread(img_path)

assert img is not None

corrected_skew = deskew(img)
grayscale_img = grayscale(corrected_skew)
inverted = invert_img(grayscale_img)
thicker = thicker_text(inverted)
# write_img('generated/final.jpg',thicker)

result = reader.readtext('generated/final.jpg')
print(result)


# display_true_img_size(img_path)
