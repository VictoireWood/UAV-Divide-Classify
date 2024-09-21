import glob
from os import listdir
import os
import copy
from pickle import FALSE

from matplotlib.scale import scale_factory
import haversine
from haversine import haversine, Unit
import numpy
import cv2
from fractions import Fraction
from tqdm import tqdm, trange
import sys
import platform
import math

import glob
from os import listdir
import os
import copy
from pickle import FALSE

from matplotlib.scale import scale_factory
import haversine
from haversine import haversine, Unit
import numpy
import cv2
from fractions import Fraction
from tqdm import tqdm, trange
import sys
import platform
import math

import random
import pandas as pd
import utm

# TODO: 
# 分辨率
resolution_w = 2048
resolution_h = 1536
# 焦距
focal_length = 1200  # TODO: the intrinsics of the camera

if platform.system() == "Windows":
    slash = '\\'
else:
    slash = '/'

basedir = '/root/shared-storage/shaoxingyu/workspace_backup/QDRaw/'
# basedir = '/root/workspace/QDRaw/'



map_dirs = {
        "2013": rf"{basedir}201310{slash}@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg",  
        "2017": rf"{basedir}201710{slash}@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg",
        "2019": rf"{basedir}201911{slash}@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg",
        "2020": rf"{basedir}202002{slash}@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg",  
        "2022": rf"{basedir}202202{slash}@map@120.42118549346924@36.60643328438966@120.4841423034668@36.573836401969416@.jpg"
    }

def photo_area_meters(flight_height):
    # 默认width更长
    # # 分辨率
    # resolution_h = 1536
    # resolution_w = 2048
    # # 焦距
    # focal_length = 1200  # TODO: the intrinsics of the camera
    map_tile_meters_w = resolution_w / focal_length * flight_height   # 相机内参矩阵里focal_length的单位是像素
    map_tile_meters_h = resolution_h / focal_length * flight_height # NOTE w768*h576
    return map_tile_meters_h, map_tile_meters_w


## 青岛位于S51区，51区中心经度为123

def crop_rot_img_wo_border(image, crop_width, crop_height, crop_center_x, crop_center_y, angle):
    if angle == 0:
        x1 = int(crop_center_x - crop_width / 2)
        y1 = int(crop_center_y - crop_height / 2)
        x2 = int(crop_center_x + crop_width / 2)
        y2 = int(crop_center_y + crop_height / 2)
        result = image[y1:y2, x1:x2]
        return image

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

def generate_map_tiles(raw_map_path:str, iter_num:int, patches_save_dir:str):

    target_w = 480
    w_h_factor = resolution_w / resolution_h
    target_h = round(target_w / w_h_factor)         # NOTE 最后要resize的高度(h360,w480)

    map_data = cv2.imread(raw_map_path)
    map_w = map_data.shape[1]   # 大地图像素宽度
    map_h = map_data.shape[0]   # 大地图像素高度

    gnss_data = raw_map_path.split('\\/')[-1]

    year_str = list (map_dirs.keys()) [list(map_dirs.values()).index(raw_map_path)]
    # year = int(year_str)

    image_save_dir = os.path.join(patches_save_dir, year_str)

    LT_lon = float(gnss_data.split('@')[2]) # left top 左上
    LT_lat = float(gnss_data.split('@')[3])
    RB_lon = float(gnss_data.split('@')[4]) # right bottom 右下
    RB_lat = float(gnss_data.split('@')[5])

    height_range = [100, 200]

    mid_lat = (LT_lat + RB_lat) / 2
    mid_lon = (LT_lon + RB_lon) / 2
    lon_res = (RB_lon - LT_lon) / map_w
    lat_res = (RB_lat - LT_lat) / map_h

    # header = pd.DataFrame(columns=['year', 'origin_img', 'flight_height', 'rotation_angle', 'zone_id', 'zone_num', 'utm_e', 'utm_n', 'loc_x', 'loc_y'])
    # csv_dir = patches_save_dir + f'{slash}Dataframes'
    # if not os.path.exists(csv_dir):
    #     os.makedirs(csv_dir)
    # csv_dataframe = csv_dir + f'{slash}{year_str}.csv'
    # header.to_csv(csv_dataframe, mode='w', index=False, header=True)

    count = 0
    with trange(iter_num, desc=year_str) as tbar:
        while count <= iter_num:
            flight_height = random.uniform(height_range[0], height_range[1])
            flight_height = round(flight_height, 2) # 取小数点后两位
            alpha = random.uniform(0, 360)
            alpha = round(alpha, 2)
            loc_w = random.randint(0, map_w)
            loc_x = loc_w
            loc_h = random.randint(0, map_h)
            loc_y = loc_h
            
            photo_meters_h, photo_meters_w = photo_area_meters(flight_height)

            map_width_meters = haversine((mid_lat, LT_lon), (mid_lat, RB_lon), unit=Unit.METERS)
            map_height_meters = haversine((LT_lat, mid_lon), (RB_lat, mid_lon), unit=Unit.METERS)
            pixel_per_meter_factor = ((map_w / map_width_meters) + (map_h / map_height_meters)) / 2     # 得出来是像素/米，每米对应多少像素
            img_h = round(photo_meters_h * pixel_per_meter_factor)
            img_w = round(photo_meters_w * pixel_per_meter_factor)
            img_lat = (loc_y + img_h / 2) * lat_res + LT_lat
            img_lon = (loc_x + img_w / 2) * lon_res + LT_lon

            utm_e, utm_n, utm_zone_num, utm_zone_id = utm.from_latlon(img_lat, img_lon)
            
            img_seg_pad = crop_rot_img_wo_border(map_data, img_w, img_h, loc_w, loc_h, alpha)

            filename = f'@{alpha:.2f}@{flight_height:.2f}@{utm_e}@{utm_n}@.png'
            # image_save_dir = patches_save_dir + f'{slash}Images{slash}{year_str}'
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            save_file_path = image_save_dir + f'{slash}{filename}'
            if os.path.exists(save_file_path):
                continue

            if img_seg_pad is None:
                pass
            else:
                count += 1
                img_seg_pad = cv2.resize(img_seg_pad, (target_w, target_h), interpolation = cv2.INTER_LINEAR)
                
                cv2.imwrite(save_file_path, img_seg_pad)
                # data_line = pd.DataFrame([[year_str, raw_map_path, flight_height, alpha, utm_zone_id, utm_zone_num, utm_e, utm_n, loc_w, loc_h]], columns=['year', 'origin_img', 'flight_height', 'rotation_angle', 'zone_id', 'zone_num', 'utm_e', 'utm_n', 'loc_x', 'loc_y'])
                # data_line.to_csv(csv_dataframe, mode='a', index=False, header=False)

            tbar.set_postfix(rate=count/iter_num, tiles=count)
            tbar.update()
        

if __name__ == '__main__':

    
    # stage = "train"

    
    # train
    # patches_save_root_dir = f'/root/shared-storage/shaoxingyu/workspace_backup/gsvqddb_{stage}/'
    # patches_save_root_dir = f'/root/workspace/gsvqddb_{stage}/'

    patches_save_root_dir = '/root/workspace/crikff47v38s73fnfgdg/backup/fake_test/'

    times = 300
    

    total_iterations = len(map_dirs)  # Total iterations  
    current_iteration = 0  # To keep track of progress  
    
    for year, map_dir in map_dirs.items():


        if not os.path.exists(patches_save_root_dir):  
            os.makedirs(patches_save_root_dir) 

        print(f"Saving tiles to: {patches_save_root_dir} ")  
        generate_map_tiles(map_dir, times, patches_save_root_dir)
            
            
        
        current_iteration += 1  # Increment the progress counter  

            # Calculate and display progress  
        progress = (current_iteration / total_iterations) * 100  
        print(f"[Progress] {progress:.2f}% complete")  