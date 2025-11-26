from glob import glob
from os import listdir
import os
import copy
from pickle import FALSE

import haversine
from haversine import haversine, Unit
import numpy as np
import cv2
from fractions import Fraction
from tqdm import tqdm, trange
import sys
import platform
import math
import utm
import pandas as pd


# TODO: 
# 分辨率
resolution_w = 2048
resolution_h = 1536
# 焦距
focal_length = 1200  # TODO: the intrinsics of the camera

def photo_area_meters(flight_height):
    # 默认width更长
    map_tile_meters_w = resolution_w / focal_length * flight_height   # 相机内参矩阵里focal_length的单位是像素
    map_tile_meters_h = resolution_h / focal_length * flight_height # NOTE w768*h576
    return map_tile_meters_h, map_tile_meters_w

if platform.system() == 'Windows':
    slash = '\\'
else:
    slash = '/'

def crop_rot_img_wo_border(image, crop_width, crop_height, crop_center_x, crop_center_y, angle):
    # 裁剪并旋转图像
    half_crop_width = (crop_width / 2)
    half_crop_height = (crop_height / 2)
    # 矩形四个顶点的坐标
    x1, y1 = crop_center_x - half_crop_width, crop_center_y - half_crop_height  # 顶点A的坐标
    x2, y2 = crop_center_x - half_crop_width, crop_center_y + half_crop_height  # 顶点B的坐标
    x3, y3 = crop_center_x + half_crop_width, crop_center_y + half_crop_height  # 顶点C的坐标
    x4, y4 = crop_center_x + half_crop_width, crop_center_y - half_crop_height  # 顶点D的坐标

    # 矩形中心点坐标
    Ox = (x1 + x2 + x3 + x4) / 4
    Oy = (y1 + y2 + y3 + y4) / 4

    # 角度转换为弧度
    alpha_rad = angle * math.pi / 180

    # 旋转矩阵
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)

    # 计算新坐标
    def rotate_point(x, y, Ox, Oy, cos_alpha, sin_alpha):
        return (
            Ox + (x - Ox) * cos_alpha - (y - Oy) * sin_alpha,
            Oy + (x - Ox) * sin_alpha + (y - Oy) * cos_alpha
        )

    # 新的四个顶点坐标
    new_x1, new_y1 = rotate_point(x1, y1, Ox, Oy, cos_alpha, sin_alpha)
    new_x2, new_y2 = rotate_point(x2, y2, Ox, Oy, cos_alpha, sin_alpha)
    new_x3, new_y3 = rotate_point(x3, y3, Ox, Oy, cos_alpha, sin_alpha)
    new_x4, new_y4 = rotate_point(x4, y4, Ox, Oy, cos_alpha, sin_alpha)
    start_x = int(min((new_x1, new_x2, new_x3, new_x4)))
    end_x = int(max((new_x1, new_x2, new_x3, new_x4)))
    start_y = int(min((new_y1, new_y2, new_y3, new_y4)))
    end_y = int(max((new_y1, new_y2, new_y3, new_y4)))

    if start_x < 0 or start_y < 0:
        return None
    elif end_x > image.shape[1] or end_y > image.shape[0]:
        return None
    else:
        cropped_image = image[start_y:end_y, start_x:end_x]

    def rotate_image(image, angle, new_w, new_h):
        (h, w) = image.shape[:2]
        (cx, cy) = (w // 2, h // 2)

        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # 调整旋转矩阵的平移部分
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy

        # 执行旋转并返回新图像
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated

    result = rotate_image(cropped_image, angle, crop_width, crop_height)
    return result

def generate_map_tiles(raw_map_path:str, patches_save_dir:str, utm_coord:tuple, filename:str):

    utm_e = utm_coord[0]
    utm_n = utm_coord[1]

    photo_lat, photo_lon = utm.to_latlon(utm_e, utm_n, 51,'S')

    # flight_height = photo_alt - 35
    photo_alt = 125
    flight_height = photo_alt

    # 飞行高度
    # flight_height = 150
    # flight_height = int(patches_save_dir.split('_')[-1])

    #TODO: 
    target_w = 2048                  # TODO: set the width corresponding to the shape of your query image
    w_h_factor = resolution_w / resolution_h
    target_h = round(target_w / w_h_factor)         # NOTE 最后要resize的高度(h360,w480)

    target_h = 1536


    # target_h = 360
    # w_h_factor = target_h / target_w
    # map_tile_width_meters = 300     # TODO: set the meters in width the you want to crop

    # 这是指地图切片的宽度对应到地面上是多长（米为单位）
    # map_tile_heigth_meters = map_tile_width_meters * w_h_factor


    # 对应地面宽高（像素为单位）
    #LINK: https://cameraharmony.com/wp-content/uploads/2020/03/focal-length-graphic-1-2048x1078.png

    map_tile_meters_w = resolution_w / focal_length * flight_height   # 相机内参矩阵里focal_length的单位是像素
    map_tile_meters_h = resolution_h / focal_length * flight_height # NOTE w768*h576

    map_data = cv2.imread(raw_map_path)

    map_w = map_data.shape[1]   # 大地图像素宽度
    map_h = map_data.shape[0]   # 大地图像素高度

    gnss_data = raw_map_path.split(slash)[-1]

    LT_lon = float(gnss_data.split('@')[2]) # left top 左上
    LT_lat = float(gnss_data.split('@')[3])
    RB_lon = float(gnss_data.split('@')[4]) # right bottom 右下
    RB_lat = float(gnss_data.split('@')[5])

    lon_res = (RB_lon - LT_lon) / map_w     # 大地图的纬线方向每像素代表的经度跨度
    lat_res = (RB_lat - LT_lat) / map_h     # 大地图的经线方向每像素代表的纬度跨度

    # map_width_meters = abs(LT_e - RB_e)
    # map_height_meters = abs(LT_n - RB_n)

    mid_lat = (LT_lat + RB_lat) / 2
    mid_lon = (LT_lon + RB_lon) / 2

    map_width_meters = haversine((mid_lat, LT_lon), (mid_lat, RB_lon), unit=Unit.METERS)
    map_height_meters = haversine((LT_lat, mid_lon), (RB_lat, mid_lon), unit=Unit.METERS)

    # pixel_per_meter_factor_w = map_w / map_width_meters
    # pixel_per_meter_factor_h = map_h / map_height_meters
    pixel_per_meter_factor = ((map_w / map_width_meters) + (map_h / map_height_meters)) / 2     # 得出来是像素/米，每米对应多少像素

    img_w = round(pixel_per_meter_factor * map_tile_meters_w)
    img_h = round(pixel_per_meter_factor * map_tile_meters_h)

    photo_mid_pixels_w = (photo_lon - LT_lon) / lon_res
    photo_mid_pixels_h = (photo_lat - LT_lat) / lat_res

    crop_center_x = round(photo_mid_pixels_w)
    crop_center_y = round(photo_mid_pixels_h)

    photo_origin_pixels_w = round(pixel_per_meter_factor * map_tile_meters_w)
    photo_origin_pixels_h = round(pixel_per_meter_factor * map_tile_meters_h)

    photo_angle = 0

    img_seg_pad = crop_rot_img_wo_border(map_data, photo_origin_pixels_w, photo_origin_pixels_h, crop_center_x, crop_center_y, photo_angle)

    photo_filename = f'@{photo_lon}@{photo_lat}@{utm_e}@{utm_n}@.png'
    filename = os.path.basename(image_path).replace('.png', '')
    new_filename = f'@ct' + filename + f'@{utm_e}@{utm_n}@.png'
    photo_filename = new_filename

    save_file_path = os.path.join(patches_save_dir, photo_filename)

    cv2.imwrite(save_file_path, img_seg_pad)

def get_utm_coord(images_paths):
    images_metadatas = [p.split("@") for p in images_paths]
    # field 1 is UTM east, field 2 is UTM north
    # self.utmeast_utmnorth = np.array([(m[1], m[2]) for m in images_metadatas]).astype(np.float64)   # ANCHOR: 原始
    # self.utmeast_utmnorth = np.array([(m[-3], m[-2]) for m in images_metadatas]).astype(np.float64)   # REVIEW: 邵星雨改，我设置的图片格式是@角度（默认0）@UTM-east@UTM-north@.png
    utmeast_utmnorth = []
    for m in images_metadatas:
        utmeast_str = m[-3]
        utmnorth_str = m[-2]
        utmeast_float64 = float(utmeast_str)
        utmnorth_float64 = float(utmnorth_str)
        utmeast_utmnorth_m = (utmeast_float64, utmnorth_float64)
        utmeast_utmnorth.append(utmeast_utmnorth_m)

        # utmeast_utmnorth = np.array([(m[-3], m[-2]) for m in images_metadatas]).astype(np.float64)   # EDIT: 邵星雨改，我设置的图片格式是@角度（默认0）@高度@UTM-east@UTM-north@.png
    # utmeast_utmnorth = np.array([(m[-3], m[-2])] for m in images_metadatas).astype(np.float64)
    # utmeast_utmnorth = np.array(utmeast_utmnorth)
    return utmeast_utmnorth

if __name__ == '__main__':
    map_path = r'/root/workspace/ctf53sc7v38s73e0mksg/maps/new_qd_years/@20250117@120.40747060326400@36.60902770136900@120.46053192297900@36.58563292439400@.tif'
    dir = r'/root/workspace/ctf53sc7v38s73e0mksg/maps/qd-test-125'
    base_dir = r'/root/workspace/ctf53sc7v38s73e0mksg/code/UAV-Divide-Classify/tmp_img/qd_test_cut'

    images_paths = sorted(glob(f"{base_dir}/**/*.png", recursive=True))

    utm_list = get_utm_coord(images_paths)

    # utm_coord = (268762.8322741015, 4052909.889267136)
    # utm_coord = (268328.9976074348, 4052584.5132671357)
    # utm_coord = (268328.9976074348, 4053235.265267136)
    # utm_coord = (269196.6669407682, 4053235.265267136)
    # utm_coord = (269281.5173397323, 4052862.478401827)
    # utm_coord = (270773.6739394539, 4052727.24788046)
    # utm_coord = (270713.77612471144, 4052732.7434378015)
    # utm_coord = (268972.70039444807, 4052446.244878582)
    # utm_coord = (270832.81511155475, 4052721.9172736183)
    # utm_coord = (270988.4401538855, 4052707.752495204)
    # dir = r'C:\Users\stara\Downloads'
    # for utm_coord in utm_list:

    os.makedirs(dir, exist_ok=True)
    tbar = tqdm(total=len(images_paths), position=0, leave=True)

    for i in range(len(images_paths)):
        utm_coord = utm_list[i]
        image_path = images_paths[i]
        filename = os.path.basename(image_path)
    
        generate_map_tiles(map_path, dir, utm_coord, filename)
        tbar.update(1)