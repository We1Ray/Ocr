import os
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def ocr_large_print_image(processor, model, src_img):
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def trocr(route):
    large_print_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    large_print_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
    shoes_bot = Image.open(route).convert("RGB")
    print(ocr_large_print_image(large_print_processor, large_print_model, shoes_bot))
    shoes_bot.close()


if __name__ == "__main__":
    # trocr('video-img/crop/1709277328.0140798.jpg_0.jpg')
    large_print_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    large_print_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
    for filename in os.listdir("video-img/crop"):
        image = Image.open(f"video-img/crop/{filename}")
        image_np = np.array(image)
        image.close()
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # 全域
        enhanced_image = cv2.equalizeHist(gray_image)
        # 區域
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(30, 10))
        enhanced_image = clahe.apply(enhanced_image)
        enhanced_image_color = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

        print(ocr_large_print_image(large_print_processor, large_print_model, enhanced_image_color))
