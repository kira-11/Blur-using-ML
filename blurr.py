import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import resize

base_options = python.BaseOptions(model_asset_path='G:\\My Drive\\codes on desktop\\VirtualTryOn\\deeplabv3.tflite')#this is a sample, use your own path
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

# Blur the image background based on the segmentation mask.
def blur_image(IMAGE_FILENAMES):
    # Create the segmenter
    with python.vision.ImageSegmenter.create_from_options(options) as segmenter:

    # Loop through available image(s)
        for image_file_name in IMAGE_FILENAMES:

            # Create the MediaPipe Image
            image = mp.Image.create_from_file(image_file_name)

            # Retrieve the category masks for the image
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask

            # Convert the BGR image to RGB
            image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

            # Apply effects
            blurred_image = cv2.GaussianBlur(image_data, (55,55), 0)
            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
            output_image = np.where(condition, image_data, blurred_image)

            print(f'Blurred background of {image_file_name}:')
            resize.resize_and_show(output_image)
