import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import cv2
import numpy as np
import math
from control.abstractions import *




class FindMask:
    def __init__(self, frame=None, contours = None, color = None):
        self.frame = frame
        self.contours = contours
        self.color = color
        self.cntr_frame = Point(frame.shape[1::-1]) / 2
        self.zeros_frame = np.zeros(frame.shape, np.uint8)
        self.zeros_bit = np.zeros([frame.shape[0], frame.shape[1], 1], np.uint8)

    def normalize(self):
        self.frame = cv2.normalize(self.frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return

    def compactness(self):
        a = []
        for c in self.contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0 or area == 0:
                return 0
            compactness = (4 * math.pi * area) / (perimeter * perimeter)
            a.append(compactness)
        return a

    def approx(self, k=0.02):
        a = []
        for cnt in self.contours:
            perimeter = cv2.arcLength(cnt, True)
            epsilon = perimeter * k
            approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
            a.append(approx_cnt)
        self.contours = np.array(a)
        return approx_cnt

    def HSV2Gray(self, Hk, Sk, Vk):
        img = self.frame
        h = img[:, :, 0].astype(np.float32)
        s = img[:, :, 1].astype(np.float32)
        v = img[:, :, 2].astype(np.float32)
        gray_float = (h * Hk) / 3 + (s * Sk) / 3 + (v * Vk) / 3
        gray = np.clip(gray_float, 0, 255).astype(np.uint8)
        self.frame = gray
        return gray

    def getCenter(self):
        a = []
        for contour in self.contours:
            m = cv2.moments(contour)
            if m['m00'] > 0:
                color_cntr = Point((m["m10"] / m["m00"], m["m01"] / m["m00"]))
                a.append(color_cntr)
        return a

    def inRangeF(self, color=None):
        """Возвращает бинарную маску для заданного цвета."""
        try:
            img = self.frame
            lower = (int(color['h_min']), int(color['s_min']), int(color['v_min']))
            upper = (int(color['h_max']), int(color['s_max']), int(color['v_max']))
            masked = cv2.inRange(img, lower, upper)
            # Обрезка верхней части
            obrez = int(color['obrez'])
            if obrez > 0:
                masked[:obrez, :] = 0
            self.frame = masked
            return masked
        except:
            self.frame = self.zeros_bit.copy()
            return self.frame

    def sortedContours(self):
        contours = sorted(self.contours, key=cv2.contourArea, reverse=True)
        self.contours = contours
        return contours

    def findContours(self):

        contours, _ = cv2.findContours(self.frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        self.contours = contours
        return contours
