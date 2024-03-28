import math
import os
import time
import cv2
import numpy as np
from mmocr.apis import MMOCRInferencer, TextDetInferencer
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from skimage.color import rgb2gray
from deskew import determine_skew
import uuid


def canny_huff(img):
    # enhanced_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # enhanced_image = cv2.equalizeHist(enhanced_image)

    edges = cv2.Canny(img, 160, 180, apertureSize=3)
    # 对边缘检测结果应用霍夫变换
    lines = cv2.HoughLines(edges, 3, np.pi / 180, 300)
    angle = []
    # 分析霍夫变换的输出
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # 计算直线的角度
            angle.append(np.rad2deg(theta))

    arr = np.array(angle)
    # 过滤数组，排除大于165和小于15的值
    filtered_arr = arr[(arr < 165) & (arr > 15)]
    # filtered_arr = arr
    # print(filtered_arr)
    # 如果过滤后的数组为空，则表示原始数组中没有符合条件的值
    if len(filtered_arr) == 0:
        median = np.average(arr)
    else:
        median = np.average(filtered_arr)
    return 0 if math.isnan(median) else round(median)


def skew_angle(img):
    grayscale = rgb2gray(img)
    angle = math.ceil(determine_skew(grayscale, sigma=2.5))
    # return angle
    # print(angle)
    if angle > 0:
        if 30 > angle > 15:
            return angle + 10
        elif angle < 15:
            return 0
        else:
            return angle - 95
    else:
        if abs(angle) > 45 or abs(angle) < 15:
            return 0
        elif abs(angle) > 20:
            return angle - 10
        else:
            return angle - 20


def crop_text_regions(image, predictions):
    cropped_images = []
    for prediction in predictions:
        det_polygons = prediction['polygons']
        for det_polygon in det_polygons:
            # 將多邊形座標轉換為 NumPy 數組
            pts = np.array(det_polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # 利用多邊形座標創建 mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], (255))

            # 找到 mask 中非黑色部分的邊界框
            x, y, w, h = cv2.boundingRect(pts)

            # 使用邊界框範圍裁剪圖片
            cropped_image = image[y:y + h, x:x + w]
            cv2.imwrite(f'crop/{uuid.uuid4()}.jpg', cropped_image)
            cropped_images.append(cropped_image)
    return cropped_images


def ocr_large_print_image(processor, model, src_img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values.to(device)
    # pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values).to(device)
    # generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def mmo():
    mocr = MMOCRInferencer(
        det='TextSnake',
        rec='ABINet',
        # det='DBNetpp',
        # rec='SATRN',
    )

    for filename in os.listdir("video-img"):
        image = cv2.imread(f'video-img/{filename}')
        angle = skew_angle(image)

        scale = 1  # 缩放比例
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        # 全域
        # enhanced_image = cv2.equalizeHist(gray_image)
        # 區域
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(30, 10))
        enhanced_image = clahe.apply(gray_image)

        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'affine-image/{filename}', enhanced_image)

        mocr(enhanced_image,
             show=False,
             print_result=True,
             # out_dir='video-img/crop',
             # save_vis=True
             )


def paddle():
    pocr = PaddleOCR(use_angle_cls=True,
                     cls_thresh=0.5,
                     use_mp=True,
                     use_gpu=True,
                     det_db_thresh=0.1,
                     show_log=False,
                     lang="en")
    for filename in os.listdir("video-img"):
        image = cv2.imread(f'video-img/{filename}')

        # huff = canny_huff(image)
        # angle = 0 if abs(abs(huff) - 90) < 15 or abs(huff) < 15 else abs(huff) - 45 if abs(huff) < 45 else 45 - abs(
        #     huff)

        angle = skew_angle(image)

        scale = 1  # 缩放比例
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        #########################################################################
        gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((4, 4), np.uint8)
        # enhanced_image = cv2.blur(gray_image, 5)
        # enhanced_image = cv2.GaussianBlur(gray_image, 5, 0)
        # enhanced_image = cv2.dilate(gray_image, kernel, iterations=1)
        enhanced_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
        # enhanced_image = cv2.medianBlur(gray_image, 5)

        # 全域
        # enhanced_image = cv2.equalizeHist(enhanced_image)
        # 區域
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(30, 10))
        enhanced_image = clahe.apply(enhanced_image)

        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'affine-image/{filename}', enhanced_image)
        #####################################################################
        result = pocr.ocr(enhanced_image, cls=True)
        for idx in range(len(result)):
            result = result[0]
        if result:
            txts = [line[1][0] for line in result]
            print(txts)
        else:
            print('None')


def tr():
    mocr_text_detect = TextDetInferencer(model='TextSnake')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    large_print_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device)
    large_print_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')

    for filename in os.listdir("video-img"):
        image = cv2.imread(f'video-img/{filename}')

        # huff = canny_huff(image)
        # angle = 0 if abs(abs(huff) - 90) < 15 or abs(huff) < 15 else abs(huff) - 45 if abs(huff) < 45 else 45 - abs(
        #     huff)

        angle = skew_angle(image)

        scale = 1.1  # 缩放比例
        # 获取图像尺寸
        height, width = image.shape[:2]
        # 定义旋转中心
        center = (width // 2, height // 2)
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # 执行仿射变换（旋转）
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        # huff = canny_huff(rotated_image)
        # angle = abs(huff - 90) if abs(huff - 90) < 20 else -huff if abs(huff) < 90 else 90 - abs(huff)  # 旋转角度

        # height, width = rotated_image.shape[:2]
        # center = (width // 2, height // 2)
        # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        # rotated_image = cv2.warpAffine(rotated_image, rotation_matrix, (width, height))

        #########################################################################
        gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((4, 4), np.uint8)
        # enhanced_image = cv2.blur(gray_image, 5)
        # enhanced_image = cv2.GaussianBlur(gray_image, 5, 0)
        # enhanced_image = cv2.dilate(gray_image, kernel, iterations=1)
        enhanced_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
        # enhanced_image = cv2.medianBlur(gray_image, 5)

        # 全域
        # enhanced_image = cv2.equalizeHist(enhanced_image)
        # 區域
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(30, 10))
        enhanced_image = clahe.apply(enhanced_image)

        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'affine-image/{filename}', enhanced_image)
        #####################################################################

        detect = mocr_text_detect(enhanced_image, show=False)
        # print(detect)
        cropped_images = crop_text_regions(enhanced_image, detect['predictions'])
        for i, cropped_image in enumerate(cropped_images):
            if cropped_image.shape[0] > 1 and cropped_image.shape[1] > 1:
                value = ocr_large_print_image(large_print_processor, large_print_model, cropped_image)
                print(value)
            else:
                continue


# ['76','SPM18022-IEO.CMM-A2','9.B2']
# ['GHOST 16','BKSCU 40532','8.5BL','2023.07','GSM/YUEFA']

if __name__ == "__main__":
    start_time = time.time()
    # mmo()
    # tr()
    paddle()
    end_time = time.time()
    run_time = end_time - start_time
    print(f"gpu運行時間: {run_time} 秒")
