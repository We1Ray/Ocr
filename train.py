from paddleocr import PaddleOCR
from PIL import Image
import cv2


def p_ocr():
    img = cv2.imread('image/yyy.jpg')

    for x in range(70, 160, 15):
        for y in range(70, 160, 15):
            for z in range(70, 160, 15):
                for a in range(160, 230, 10):
                    for b in range(160, 230, 10):
                        for c in range(160, 230, 10):
                            lower = (x, y, z)
                            upper = (a, b, c)
                            # 設定顏色範圍
                            # lower_red = (180, 185, 190)
                            # upper_red = (200, 210, 215)

                            # 使用 inRange() 函數檢測顏色範圍
                            mask = cv2.inRange(img, lower, upper)

                            # 使用 threshold() 函數將顏色範圍轉換成黑色
                            img[mask > 0] = (150, 150, 150)

                            # 顯示圖片
                            # cv2.imshow(img)
                            # cv2.waitKey(0)

                            black = Image.fromarray(img)
                            black.save('image/a.jpg')

                            # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
                            # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
                            ocr = PaddleOCR(use_angle_cls=True,
                                            lang="en")  # need to run only once to download and load model into memory
                            img_path = 'image/a.jpg'
                            result = ocr.ocr(img_path, cls=True)
                            for idx in range(len(result)):
                                res = result[idx]
                                # for line in res:
                                #     print(line)
                                # 显示结果
                                result = result[0]
                            if result:
                                image = Image.open(img_path).convert('RGB')
                                boxes = [line[0] for line in result]
                                txts = [line[1][0] for line in result]
                                scores = [line[1][1] for line in result]

                                with open("output.txt", "a") as f:
                                    f.write("lower: {}, upper: {}, txt: {}\n".format(lower, upper, txts))
                                # im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
                                # im_show = Image.fromarray(im_show)
                                # im_show.save('result.jpg')
                            print(lower, upper)


if __name__ == "__main__":
    p_ocr()
