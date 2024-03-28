import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import cv2
import requests
import transformers
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from paddleOcr import p_ocr
import cv2
import pytesseract
from pytesseract import Output
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"


def ocr_large_print_image(processor, model, src_img):
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def crop(route):
    large_print_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    large_print_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
    shoes_bot = Image.open(route).convert("RGB")

    # print(shoes_bot.size[0])
    # print(shoes_bot.size[1])

    # shoes_bot_crop = shoes_bot.crop((0, 0, shoes_bot.size[0], shoes_bot.size[1]))

    # shoes_bot_crop = shoes_bot.crop((400, 1550, 2250, 2900))

    shoes_bot_crop = shoes_bot.crop((600, 2100, 2250, 2600))
    # shoes_bot_crop = shoes_bot.crop((400, 2600, 2250, 2900))
    shoes_bot_crop.save(f'crop-image/1.jpg')
    print(ocr_large_print_image(large_print_processor, large_print_model, shoes_bot_crop))

    # img = cv2.cvtColor(np.array(shoes_bot), cv2.COLOR_RGB2BGR)
    # mask = cv2.inRange(img, (0, 0, 0), (130, 130, 130))
    # img[mask > 0] = (255, 255, 255)
    # mask = cv2.inRange(img, (135, 135, 130), (165, 165, 145))
    # img[mask > 0] = (100, 100, 100)
    # Image.fromarray(img).show()

    # for i in range(0, 1100, 50):
    #     shoes_bot_crop2 = shoes_bot_crop.crop((0, i, shoes_bot_crop.size[0], i + 350))

    # img = cv2.imread(f'crop-image/{i}.jpg')

    # img = cv2.cvtColor(np.array(shoes_bot_crop2), cv2.COLOR_RGB2BGR)

    # mask = cv2.inRange(img, (0, 0, 0), (130, 130, 130))
    # img[mask > 0] = (255, 255, 255)
    #
    # mask = cv2.inRange(img, (130, 130, 130), (140, 140, 140))
    # img[mask > 0] = (130, 130, 130)
    #
    # mask = cv2.inRange(img, (140, 140, 140), (150, 150, 150))
    # img[mask > 0] = (140, 140, 140)

    # img = Image.fromarray(img)

    # print(i)
    # print(ocr_large_print_image(large_print_processor, large_print_model, shoes_bot_crop2))
    #
    # shoes_bot_crop2.save(f'crop-image/{i}.jpg')
    # shoes_bot_crop2.close()
    # p_ocr(f'crop-image/{i}.jpg')
    # print('---------------------------------')
    shoes_bot.close()


def black_white(route):
    image = cv2.imread(route)
    # 黑白
    _, binary_image = cv2.threshold(image, 128, 180, cv2.THRESH_BINARY)

    image = Image.fromarray(binary_image)

    p_ocr(binary_image)
    image.save(f'change/black_white_image.jpg')
    image.close()


def rgb_change(route):
    image = cv2.imread(route)

    # 依RGB範圍調色
    mask = cv2.inRange(image, (0, 0, 0), (115, 115, 115))
    image[mask > 0] = (150, 150, 150)

    # condition = (img < 170) & (img > 130)
    # img[condition] -= 10

    # mask = cv2.inRange(img, (150, 150, 150), (255, 255, 255))
    # img[mask > 0] = (230, 230, 230)
    image = Image.fromarray(image)
    p_ocr(image)
    image.save(f'change/rgb_image.jpg')
    image.close()


def pyc(route):
    image = cv2.imread(route)

    d = pytesseract.image_to_data(image, output_type=Output.DICT)

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    p_ocr(image)
    image = Image.fromarray(image)
    image.save(f'change/pyc_image.jpg')
    image.close()


# 對全局做直方圖均衡化
def equalize(route):
    img = cv2.imread(route)

    # convert the image into grayscale before doing histogram equalization
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # image equalization
    equalize_img = cv2.equalizeHist(gray_img)
    p_ocr(equalize_img)

    image = Image.fromarray(equalize_img)
    image.save(f'change/equalize_image.jpg')


def dilated_mask(route):
    upper = np.array([120, 191, 0])
    lower = np.array([124, 198, 0])
    image = cv2.imread(route)
    # 遮罩
    mask = cv2.inRange(image, lower, upper)

    if cv2.countNonZero(mask) == 0:
        print("沒有找到符合像素")
    else:
        # 增加強度
        darkened = image.copy()
        kernel_size = 5  # 膨脹程度
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        darkened[dilated_mask != 0] = [0, 0, 0]
        # for i in range(3):  # 分別操作
        # darkened[:, :, i] = np.where(mask != 0, darkened[:, :, i] * 0.5, darkened[:, :, i])
        # darkened[mask != 0] = [0, 0, 0]
        darkened[dilated_mask != 0] = [0, 0, 0]
        result = cv2.addWeighted(image, 0.5, darkened, 0.5, 0)
        p_ocr(result)
        image = Image.fromarray(result)
        image.save('change/dilated_mask_image.jpg')


def change(route):
    image = cv2.imread(route)

    # 二值化
    # _, binary_image = cv2.threshold(image, 100, 180, cv2.THRESH_BINARY)
    # binary_image = Image.fromarray(binary_image)
    # binary_image.save('change/binary_image.jpg')

    # 去噪
    # denoised_image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # denoised_image = Image.fromarray(denoised_image)
    # denoised_image.save('change/denoised_image.jpg')

    # 增強對比度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for i in np.arange(0.5, 5, 0.5):
        for j in np.arange(1, 100, 1):
            for k in np.arange(1, 100, 1):
                clahe = cv2.createCLAHE(clipLimit=i, tileGridSize=(j, k))
                enhanced_image = clahe.apply(gray_image)

                enhanced_image = Image.fromarray(enhanced_image)
                enhanced_image.save('change/enhanced_image.jpg')
                enhanced_image.close()

                with open("output.txt", "a") as f:
                    f.write(
                        "clipLimit: {}, upper: ({},{}), txt: {}\n".format(i, j, k, p_ocr('change/enhanced_image.jpg')))

    # 對區域做直方圖均衡化
    # clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(80, 18))
    # enhanced_image = clahe.apply(gray_image)
    # enhanced_image = Image.fromarray(enhanced_image)
    # enhanced_image.save('change/enhanced_image.jpg')
    # enhanced_image.close()
    # p_ocr('change/enhanced_image.jpg')

    # 調整亮度和對比度
    # alpha = 1.5  # 亮度增強因子
    # beta = 10  # 對比度增強因子
    # adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # adjusted_image = Image.fromarray(adjusted_image)
    # adjusted_image.save('change/adjusted_image.jpg')

    # 銳化
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # sharpened_image = cv2.filter2D(image, -1, kernel)
    # sharpened_image = Image.fromarray(sharpened_image)
    # sharpened_image.save('change/sharpened_image.jpg')


if __name__ == "__main__":
    # p_ocr('change/equalize_image.jpg')
    # crop('change/enhanced_image.jpg')
    # rgb_change('image/S__126148614_0.jpg')
    # dilated_mask('image/abc.jpg')
    # pyc('image/S__126148614_0.jpg')
    # change('image/S__126148614_0.jpg')
    black_white('image/abc.jpg')
    equalize('image/abc.jpg')
