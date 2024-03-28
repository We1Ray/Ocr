import os
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import cv2
import easyocr


# edges = cv2.Canny(img, 160, 180, apertureSize=3) #控制會不會旋轉
# lines = cv2.HoughLines(edges, 3, np.pi / 180, 88) # 控制旋轉幅度

def p_ocr(route):
    # img = cv2.imread(route)

    # lower = (0, 0, 0)
    # upper = (196, 196, 196)
    # # 使用 inRange() 函數檢測顏色範圍
    # mask = cv2.inRange(img, lower, upper)
    # # 使用 threshold() 函數將顏色範圍轉換成黑色
    # img[mask > 0] = (150, 150, 150)

    # 顯示圖片
    # cv2.imshow(img)
    # cv2.waitKey(0)

    # black = Image.fromarray(img)
    # black.save('image/b.jpg')

    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True,
                    use_gpu=True,
                    lang="en")  # need to run only once to download and load model into memory
    # img_path = 'crop-image/1.png'
    result = ocr.ocr(route, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        # for line in res:
        #     print(line)
        # 显示结果
        result = result[0]
    if result:
        # image = Image.open(route).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        # im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
        # im_show = Image.fromarray(im_show)
        # im_show.save('result.jpg')
        print(txts)
        return txts


def easy_ocr(route):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(route)
    for i in result:
        print(i)


def pp():
    ocr = PaddleOCR(use_angle_cls=True,
                    use_gpu=True,
                    # save_crop_res=True,
                    # crop_res_save_dir="video-img/crop",
                    lang="en")
    for filename in os.listdir("video-img"):
        image = Image.open(f'video-img/{filename}')
        image_np = np.array(image)
        image.close()
        # result = ocr.ocr(image_np, cls=True)

        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # 全域
        enhanced_image = cv2.equalizeHist(gray_image)

        # 區域
        # clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(30, 10))
        # enhanced_image = clahe.apply(enhanced_image)

        result = ocr.ocr(enhanced_image, cls=True)
        for idx in range(len(result)):
            result = result[0]
        if result:
            txts = [line[1][0] for line in result]
            print(txts)


if __name__ == "__main__":
    # p_ocr('image/537649.jpg')
    # easy_ocr('change/enhanced_image.jpg')
    pp()
