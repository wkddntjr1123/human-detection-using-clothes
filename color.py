import cv2
import numpy as np

# bgr
table = {
    "red": [np.array([165, 100, 20], np.uint8), np.array([179, 255, 255], np.uint8)],  # 빨강
    "orange": [np.array([10, 60, 20], np.uint8), np.array([24, 255, 255], np.uint8)],  # 주황
    "yellow": [np.array([22, 60, 20], np.uint8), np.array([33, 255, 255], np.uint8)],  # 노랑
    "blue": [np.array([110, 100, 20], np.uint8), np.array([130, 255, 255], np.uint8)],  # 파랑
    "brown": [np.array([10, 120, 50], np.uint8), np.array([23, 255, 255], np.uint8)],  # 갈색
    "indigo": [np.array([105, 60, 38], np.uint8), np.array([140, 210, 85], np.uint8)],  # 남색
    "pink": [np.array([120, 50, 20], np.uint8), np.array([173, 215, 255], np.uint8)],  # 핑크
    "black": [np.array([0, 0, 0], np.uint8), np.array([179, 255, 30], np.uint8)],  # 검정
    "white": [np.array([0, 0, 231], np.uint8), np.array([180, 18, 255], np.uint8)],  # 하양
    "beige": [np.array([0, 5, 20], np.uint8), np.array([255, 20, 255], np.uint8)],  # 베이지
    "sky": [np.array([88, 40, 20], np.uint8), np.array([110, 255, 255], np.uint8)],  # 하늘
    "purple": [np.array([135, 150, 20], np.uint8), np.array([158, 255, 255], np.uint8)],  # 보라
    "mauve": [np.array([145, 60, 20], np.uint8), np.array([165, 255, 255], np.uint8)],  # 연보라
    "green": [np.array([38, 40, 20], np.uint8), np.array([73, 255, 255], np.uint8)],  # 초록
    "gray": [np.array([0, 0, 40], np.uint8), np.array([180, 18, 230], np.uint8)],  # 회색
}
red_lower = []


def check_color(
    hsv_frame, is_top, is_bot, input_top_color, input_bot_color, map_result, x1, y1, x2, y2
):
    total_top_cnt = top_cnt = total_bot_cnt = bot_cnt = 0
    top_threshed = cv2.inRange(hsv_frame, table[input_top_color][0], table[input_top_color][1])
    bot_threshed = cv2.inRange(hsv_frame, table[input_bot_color][0], table[input_bot_color][1])
    # cv2.imwrite("orin_red.png",hsv_frame)
    # cv2.imwrite("mask_red.png",top_threshed)
    for x in range(x1, x2):
        for y in range(y1, y2):
            if map_result[y][x] == 2:  # 상의
                total_top_cnt += 1
                if top_threshed[y][x]:  # 마스크결과가 있으면
                    top_cnt += 1
            elif map_result[y][x] == 5:  # 하의
                total_bot_cnt += 1
                if bot_threshed[y][x]:
                    bot_cnt += 1
    if (total_top_cnt and top_cnt / total_top_cnt >= 0.2) or (
        total_bot_cnt and bot_cnt / total_bot_cnt >= 0.2
    ):
        return True
    return False
