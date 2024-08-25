import sys
import numpy as np
import cv2
import math
import argparse
import time
import os
from concurrent.futures import ThreadPoolExecutor



def radians(degrees):
    return degrees * np.pi / 180

class Equi2Rect:
    def __init__(self, pan, tilt):
        self.w = 1280
        self.h = 720
        self.yaw = radians(pan)
        self.pitch = radians(tilt)
        self.roll = radians(0.0)
        self.Rot = self.eul2rotm(self.pitch, self.yaw, self.roll)
        self.f = 800
        self.K = np.array([[self.f, 0, self.w / 2],
                           [0, self.f, self.h / 2],
                           [0, 0, 1]])
        self.img_interp = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def eul2rotm(self, rotx, roty, rotz):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(rotx), -math.sin(rotx)],
                        [0, math.sin(rotx), math.cos(rotx)]])

        R_y = np.array([[math.cos(roty), 0, math.sin(roty)],
                        [0, 1, 0],
                        [-math.sin(roty), 0, math.cos(roty)]])

        R_z = np.array([[math.cos(rotz), -math.sin(rotz), 0],
                        [math.sin(rotz), math.cos(rotz), 0],
                        [0, 0, 1]])

        R = R_z.dot(R_y).dot(R_x)
        return R

    def set_image(self, img):
        self.img_src = img
        self.img_interp = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def vectorized_reprojection(self):
        x_img, y_img = np.meshgrid(np.arange(self.w), np.arange(self.h))
        xyz = np.stack([x_img.flatten(), y_img.flatten(), np.ones_like(x_img).flatten()], axis=-1)
        xyz_norm = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

        RK = self.Rot @ np.linalg.inv(self.K)
        ray3d = RK @ xyz_norm.T

        xp, yp, zp = ray3d
        theta = np.arctan2(yp, np.sqrt(xp ** 2 + zp ** 2))
        phi = np.arctan2(xp, zp)

        x_sphere = ((phi + np.pi) * self.img_src.shape[1] / (2 * np.pi)).reshape(self.h, self.w)
        y_sphere = ((theta + np.pi / 2) * self.img_src.shape[0] / np.pi).reshape(self.h, self.w)

        return x_sphere, y_sphere


    def perform_interpolation(self):
        x_sphere, y_sphere = self.vectorized_reprojection()
        map_x = np.float32(x_sphere)
        map_y = np.float32(y_sphere)
        self.img_interp = cv2.remap(self.img_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)

def process_vr_images(input_folder, output_folder, pan, tilt):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    equi2rect = Equi2Rect(pan, tilt)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Error: Could not load image {input_path}!")
                continue

            equi2rect.set_image(img)
            equi2rect.perform_interpolation()
            cv2.imwrite(output_path, equi2rect.img_interp)
            print(f"Processed {filename}")

    print("All images processed.")




