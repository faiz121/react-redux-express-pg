import io
import re
import base64
import numpy as np
from PIL import Image
import cv2
import sys
import os
import time
import matplotlib.pyplot as plt


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def data_url_to_arr(data_url, size):
    imgstr = re.search(r'base64,(.*)', data_url).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    im = Image.open(image_bytes)

    resized_im = im.resize(size, Image.ANTIALIAS)
    arr = np.asarray(resized_im)

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

def process_image_from_file(filename):
    with Image.open(filename) as img:
        width, height = img.size
        size = (28, 28)
        img = img.resize(size)
        pix = np.array(img, dtype='uint8')
        print('pix', pix)
        print('type:', type(pix))
        print('pix shape: ', np.shape(pix))
    return pix

def array_to_np_image(arr):
    arr_shape = np.shape(arr)
    formatted_arr = np.empty([arr_shape[0], arr_shape[1], 4], dtype='uint8')
    print('formatted_Arr_shape: ', np.shape(formatted_arr))
    for row_index, row in enumerate(arr):
        for col_index, _ in enumerate(row):
            for i in range(4):
                if i == 3:
                    formatted_arr[row_index][col_index][i] = 255
                elif i == 2:
                    formatted_arr[row_index][col_index][i] = 255
                else:
                    formatted_arr[row_index][col_index][i] = 0
    print('pix', formatted_arr)
    print('type:', type(formatted_arr))
    print('formatted_arr shape: ', np.shape(formatted_arr))
    return formatted_arr

# Outputs UN-flattened array
def np_image_to_array(np_arr):
    arr_shape = np.shape(np_arr)
    result = np.empty([arr_shape[0], arr_shape[1]], dtype='float')

    for row_index, row in enumerate(np_arr):
        for col_index, _ in enumerate(row):
            pixel_arr = np_arr[row_index][col_index]
            pixel_color = np.max(pixel_arr[:3]) / 255.0
            pixel_alpha = pixel_arr[3] / 255.0
            pixel_int_value = pixel_color * pixel_alpha
            print("pixel_int_value: ", pixel_int_value)

            result[row_index][col_index] = pixel_int_value

    return result


def process_image_from_data_url(data_url, size):
    pixel_arr = data_url_to_arr(data_url, size)
    return pixel_arr

def plot_image(image):
    plt.imshow(image)
    plt.show()

data_url = "data:image/png;base64,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"

image = process_image_from_data_url(data_url, (28, 28))
print("image: ", type(image), np.shape(image))
print("yo", np_image_to_array(image))
# plot_image(image)

# image = process_image_from_file('mypainting2.png')
# plot_image(image)
