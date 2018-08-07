import io
import re
import base64
import numpy as np
from PIL import Image
import cv2
import sys
import os
import time

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def data_url_to_arr(data_url):
    imgstr = re.search(r'base64,(.*)', data_url).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    im = Image.open(image_bytes)
    arr = np.array(im)[:,:,0]
    return arr

def resize_image(numpy_array):
    img = cv2.imread(numpy_array)
    eprint("img:", img)
    res = img.resize((28, 28))
    return res

def save_image(data_url):
    img_data = data_url.split(',')[1]
    i = 0
    while os.path.exists("pythons/tmp/sample_images/sample%s.png" % i):
        i += 1
    with open("pythons/tmp/sample_images/sample%s.png" % i, "wb") as fh:
        fh.write(base64.b64decode(img_data.encode("utf-8")))
    time.sleep(1)

 # Reverses the contents of a String or IO object.
 #
 # @param [String, #filename] contents filename
 # @return [np.Array()] the numpy array of img

def process_image():
    i = 0
    while os.path.exists("sample%s.png" % i):
        i += 1
    filename = "pythons/tmp/sample_images/sample%s.png" % i
    with Image.open(filename) as img:
        width, height = img.size
        size = (28, 28)
        img = img.resize(size)
        pix = np.array(img, dtype='uint8')
        flattened_img_array = np.array([])

        for row in pix:
            for column in row:
                for i, pixel in enumerate(column):
                    if i == 2:
                        flattened_img_array = np.append(flattened_img_array, pixel - column[3])
    return flattened_img_array

# import numpy as np
# from PIL import Image
# filename = 'mypainting.png'
# with Image.open(filename) as img:
#     width, height = img.size
#     print(width)
#     size = (28, 28)
#     img = img.resize(size)
#     pix = np.array(img, dtype='uint8')
#     print("pix: ", pix)
#     print("shape: ", np.shape(pix))
#     print("yo: ", pix)
