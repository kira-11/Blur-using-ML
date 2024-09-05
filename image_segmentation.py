import cv2
import resize
import segmenter_mask
import blurr


IMAGE_FILENAMES = ['G:\\My Drive\\codes on desktop\\VirtualTryOn\\image3.jpg']#this is a sample, use your own path
print(type(IMAGE_FILENAMES))

image = cv2.imread(IMAGE_FILENAMES[0])
resize.resize_and_show(image)

segmenter_mask.mask(IMAGE_FILENAMES)

blurr.blur_image(IMAGE_FILENAMES)

cv2.destroyAllWindows()
