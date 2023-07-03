"""
-------------------------------------------
-    Author: Asadullah Dal                -
-    =============================        -
-    Company Name: AiPhile                -
-    =============================        -
-    Purpose : Youtube Channel            -
-    ============================         -
-    Link: https://youtube.com/c/aiphile  -
-------------------------------------------
"""
import cv2 as cv
import numpy as np
import os


# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (0, 165, 255)
PINK = (147, 20, 255)
INDIGO = (75, 0, 130)
VIOLET = (238, 130, 238)
GRAY = (127, 127, 127)


def read_images_from_dir(path, resize_flag=None):
    files = os.listdir(path)
    list_path = []
    img_list = []
    for file in files:
        img_path = os.path.join(path, file)
        list_path.append(img_path)
        img = cv.imread(img_path)
        if resize_flag:
            img = cv.resize(img, resize_flag, interpolation=cv.INTER_CUBIC)
        img_list.append(img)
        # cv.imshow('test', img)
        # cv.waitKey(0)
    return img_list


def rect_corners(
    image, rect_points, color, DIV=6, th=2, opacity=0.3, draw_overlay=False
):

    x, y, w, h = rect_points
    top_right_corner = np.array(
        [[x + w // DIV, y], [x, y], [x, y + h // DIV]], dtype=np.int32
    )
    cv.rectangle(image, (x, y), (x + w, y + h), color, th // 2)
    cv.polylines(image, [top_right_corner], False, color, th)
    # top left corner
    top_left_corner = np.array(
        [[(x + w) - w // DIV, y], [x + w, y], [x + w, y + h // DIV]], dtype=np.int32
    )
    cv.polylines(image, [top_left_corner], False, color, th)

    # bottom right corner
    bottom_right_corner = np.array(
        [[x + w // DIV, y + h], [x, y + h], [x, (y + h) - h // DIV]], dtype=np.int32
    )
    cv.polylines(image, [bottom_right_corner], False, color, th)

    # bottom left corner

    bottom_left_corner = np.array(
        [[x + w, (y + h) - h // DIV], [x + w, y + h], [(x + w) - w // DIV, y + h]],
        dtype=np.int32,
    )
    if draw_overlay:
        overlay = image.copy()  # coping the image
        cv.rectangle(overlay, rect_points, color, -1)
        new_img = cv.addWeighted(overlay, opacity, image, 1 - opacity, 0)
        # print(points_list)
        image = new_img

    cv.polylines(image, [bottom_left_corner], False, color, th)

    # cv.circle(image, (x, y), 4, color, 2)
    # cv.circle(image, (x + w, y), 4, (0, 255, 0), 2)
    # cv.circle(image, (x, y + h), 4, (255, 0, 0), 2)
    # cv.circle(image, (x + w, y + h), 4, (0, 0, 255), 2)
    return image


def text_with_background(
    image,
    text,
    position=(30, 30),
    fonts=cv.FONT_HERSHEY_PLAIN,
    scaling=1,
    color=(0, 255, 255),
    th=1,
    draw_corners=True,
    up=0,
):
    image_h, image_w = image.shape[:2]
    x, y = position
    y = y - up
    (w, h), p = cv.getTextSize(text, fonts, scaling, th)
    cv.rectangle(image, (x - p, y + p), (x + w + p, y - h - p), (0, 0, 0), -1)

    if draw_corners:
        rect_points = [x - p, y - h - p, w + p + p, h + p + p]
        # cv.rectangle(image, rect_points, color=(0, 255, 0), thickness=2)
        rect_corners(image, rect_points, color, th=th, DIV=4)

    cv.putText(image, text, (x, y), fonts, scaling, color, th, cv.LINE_AA)


def fill_poly_trans(image, points, color, opacity):

    list_to_np_array = np.array(points, dtype=np.int32)
    overlay = image.copy()  # coping the image
    cv.fillPoly(overlay, [list_to_np_array], color)
    new_image = cv.addWeighted(overlay, opacity, image, 1 - opacity, 0)
    # print(points_list)
    image = new_image
    # cv.polylines(image, [list_to_np_array], True, color, 1, cv.LINE_AA)
    return image


def trans_circle(image, org, radi, color, opacity):

    overlay = image.copy()  # coping the image
    cv.circle(overlay, org, radi, color, -1)
    new_image = cv.addWeighted(overlay, opacity, image, 1 - opacity, 0.1)
    # print(points_list)
    image = new_image
    # cv.polylines(image, [list_to_np_array], True, color, 1, cv.LINE_AA)
    return image
