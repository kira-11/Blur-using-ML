import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import resize

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white


# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='G:\\My Drive\\codes on desktop\\VirtualTryOn\\deeplabv3.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)


def mask(IMAGE_FILENAMES):
    # Create the image segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:

        # Loop through demo image(s)
        for image_file_name in IMAGE_FILENAMES:

            # Create the MediaPipe image file that will be segmented
            image = mp.Image.create_from_file(image_file_name)

            # Retrieve the masks for the segmented image
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask

            # Generate solid color images for showing the output segmentation mask.
            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)

            print(f'Segmentation mask of {IMAGE_FILENAMES[0]}:')
            resize.resize_and_show(output_image)
            