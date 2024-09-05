
__all__ = ['cv2_imshow', 'cv_imshow']

import cv2
from IPython import display
import PIL

def cv2_imshow(a):
  a = a.clip(0, 255).astype('uint8')
  
  if a.ndim == 3:
    if a.shape[2] == 4:
      a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    else:
      a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  display.display(PIL.Image.fromarray(a))

cv_imshow = cv2_imshow
