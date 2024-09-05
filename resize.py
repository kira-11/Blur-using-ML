import cv2
import math
import cv2_imshow
from IPython import display
from PIL import Image

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def cv2_imshow(a):
  a = a.clip(0, 255).astype('uint8')
  
  if a.ndim == 3:
    if a.shape[2] == 4:
      a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    else:
      a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  display.display(Image.fromarray(a))


def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2_imshow(img)
  cv2.imshow("image",img)
  cv2.waitKey(0)


  
