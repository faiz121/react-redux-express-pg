import io
import re
import base64
import numpy as np
from PIL import Image
import cv2
import sys

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
