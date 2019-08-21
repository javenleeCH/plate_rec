import cv2
import numpy as np


def detect_color(im_cv):
    im_cv = cv2.imread(im_cv)
    hsv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2HSV)
    # 黄牌
    yellow_lower = np.array([22, 60, 200], np.uint8)
    yellow_upper = np.array([60, 255, 255], np.uint8)
    # 蓝牌
    blue_lower = np.array([99, 100, 130], np.uint8)
    blue_upper = np.array([115, 250, 200], np.uint8)
    # 绿牌
    green_lower = np.array([50, 0, 200], np.uint8)
    green_upper = np.array([70, 70, 255], np.uint8)

    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)

    kernal = np.ones((5, 5), "uint8")

    yellow = cv2.dilate(yellow, kernal)
    res_yellow = cv2.bitwise_and(im_cv, im_cv, mask=yellow)

    blue = cv2.dilate(blue, kernal, iterations=2)
    res_blue = cv2.bitwise_and(im_cv, im_cv, mask=blue)
    # cv2.imshow("1", blue)
    # cv2.waitKey(0)
    # cv2.imshow("2", res_blue)
    # cv2.waitKey(0)

    green = cv2.dilate(green, kernal, iterations=2)
    res_green = cv2.bitwise_and(im_cv, im_cv, mask=green)
    # cv2.imshow("1", green)
    # cv2.waitKey(0)
    # cv2.imshow("2", res_green)
    # cv2.waitKey(0)

    # yellow
    yellow_plate = []
    (contours, hierarchy)=cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(contour)
            if 3> w/h >2:
                img_y = cv2.rectangle(im_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_y, "Yellow_plate", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                print("yellow plate")
                yellow_plate.append(img_y[y:y+h, x:x+w])

    # blue
    blue_plate = []
    (contours, hierarchy) = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1600:
            x, y, w, h = cv2.boundingRect(contour)
            if 3> w/h >2:
                img_b = cv2.rectangle(im_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img_b, "blue_plate", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
                print("blue plate")
                blue_plate.append(img_b[y:y+h, x:x+w])

    # green
    green_plate = []
    (contours, hierarchy)=cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(contour)
            if 3.5> w/h >2:
                img_g = cv2.rectangle(im_cv, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(img_g, "green_plate", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                print("green plate")
                green_plate.append(img_g[y:y+h, x:x+w])

    # cv2.imshow("Color Tracking", im_cv)
    # cv2.waitKey(0)
    cv2.imwrite("s1.jpg", im_cv)
    return green_plate, blue_plate, yellow_plate


