import os
from craft_text_detector import Craft
import cv2


def craft_image():
    # set image path and export folder directory
    for filename in os.listdir("affine-image"):
        output_dir = f'craft/{filename}/'
        image = cv2.imread(f'affine-image/{filename}')

        # create a craft instance
        craft = Craft(output_dir=output_dir, crop_type="poly", cuda=True)

        # apply craft text detection and export detected regions to output directory
        try:
            prediction_result = craft.detect_text(image)
            print(prediction_result)
        except:
            continue

        # unload models from ram/gpu
        craft.unload_craftnet_model()
        craft.unload_refinenet_model()


if __name__ == "__main__":
    # can be filepath, PIL image or numpy array
    craft_image()
