import cv2
import matplotlib.pyplot as plt

des_list = []
sift = cv2.SIFT_create()
im = cv2.imread('test3.jpeg')
plt.imshow(im)


def draw_keypoints(vis, keypoints, color=(0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        plt.imshow(cv2.circle(vis, (int(x), int(y)), 2, color))

kp = sift.detect(im, None)
kp, des = sift.compute(im, kp)
draw_keypoints(im, kp)
