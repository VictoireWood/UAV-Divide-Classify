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


## 青岛位于S51区，51区中心经度为123
def S51_UTM(lon, lat):
    easting = 500000 + haversine((lat,lon), (lat, 123), unit=Unit.METERS)
    northing = haversine((lat, lon), (0, lon), unit=Unit.METERS)
    return easting, northing


def generate_map_tiles(raw_map_path:str,stride_ratio_str:str,patches_save_dir:str):
    # TODO: 
    # 分辨率
    resolution_w = 2048
    resolution_h = 1536
    # 焦距
    focal_length = 1200  # TODO: the intrinsics of the camera
    # 飞行高度
    flight_height = 450 

    #TODO: 
    target_w = 480                  # TODO: set the width corresponding to the shape of your query image
    # target_h = 360
    # w_h_factor = target_h / target_w
    # map_tile_width_meters = 300     # TODO: set the meters in width the you want to crop

    # 这是指地图切片的宽度对应到地面上是多长（米为单位）
    # map_tile_heigth_meters = map_tile_width_meters * w_h_factor


    numerator = int(stride_ratio_str.split('/')[0])
    denominator = int(stride_ratio_str.split('/')[1])
    stride_ratio = float(Fraction(numerator, denominator))

    map_data = cv2.imread(raw_map_path)

    map_w = map_data.shape[1]   # 大地图像素宽度
    map_h = map_data.shape[0]   # 大地图像素高度

    gnss_data = raw_map_path.split('\\')[-1]

    LT_lon = float(gnss_data.split('@')[2]) # left top 左上
    LT_lat = float(gnss_data.split('@')[3])
    RB_lon = float(gnss_data.split('@')[4]) # right bottom 右下
    RB_lat = float(gnss_data.split('@')[5])

    LT_e, LT_n = S51_UTM(LT_lon, LT_lat)
    RB_e, RB_n = S51_UTM(RB_lon, RB_lat)

    lon_res = (RB_lon - LT_lon) / map_w     # 大地图的纬线方向每像素代表的经度跨度
    lat_res = (RB_lat - LT_lat) / map_h     # 大地图的经线方向每像素代表的纬度跨度

    map_width_meters = abs(LT_e - RB_e)
    map_height_meters = abs(LT_n - RB_n)

    pixel_per_meter_factor = ((map_w / map_width_meters) + (map_h / map_height_meters)) / 2     # 得出来是像素/米，每米对应多少像素

    # 对应地面宽高（像素为单位）
    #LINK: https://cameraharmony.com/wp-content/uploads/2020/03/focal-length-graphic-1-2048x1078.png
    map_tile_meters_w = resolution_w / focal_length * flight_height   # 相机内参矩阵里focal_length的单位是像素
    map_tile_meters_h = resolution_h / focal_length * flight_height

    w_h_factor = resolution_w / resolution_h
    target_h = round(target_w / w_h_factor)         # 最后要resize的高度

    stride_x = round(pixel_per_meter_factor * map_tile_meters_w * stride_ratio)
    stride_y = round(pixel_per_meter_factor * map_tile_meters_h * stride_ratio)

    # 计算要切多少个tile
    iter_w = int((map_w - target_w) / stride_x) + 1
    iter_h = int((map_h - target_h) / stride_y) + 1
    iter_total = iter_w * iter_h

    img_w = round(pixel_per_meter_factor * map_tile_meters_w)
    img_h = round(pixel_per_meter_factor * map_tile_meters_h)

    with trange(iter_total, desc=gnss_data) as tbar:
        i = 0
        loc_x = 0
        # LINK: https://blog.csdn.net/winter2121/article/details/111356587
        while loc_x < map_w - img_w:    # 已分割像素宽度<大地图宽度-地图切片宽度
            loc_y = 0
            while loc_y < map_h - img_h:
                LT_cur_lon = str(loc_x * lon_res + LT_lon)
                LT_cur_lat = str(loc_y * lat_res + LT_lat)
                RB_cur_lon = str((loc_x + img_w) * lon_res + LT_lon)
                RB_cur_lat = str((loc_y + img_h) * lat_res + LT_lat)
                CT_cur_lon = str((loc_x + img_w / 2) * lon_res + LT_lon)    # centre
                CT_cur_lat = str((loc_y + img_h / 2) * lat_res + LT_lat)
                CT_cur_lon_ = (loc_x + img_w / 2) * lon_res + LT_lon    # centre
                CT_cur_lat_ = (loc_y + img_h / 2) * lat_res + LT_lat
                CT_utm_e, CT_utm_n = S51_UTM(CT_cur_lon_, CT_cur_lat_)

                img_seg_pad = map_data[loc_y:loc_y + img_h, loc_x:loc_x + img_w]
                img_seg_pad = cv2.resize(img_seg_pad, (target_w, target_h), interpolation = cv2.INTER_LINEAR)

                tbar.set_postfix(rate=i/iter_total, tiles=i)
                tbar.update()

                # 决定是否要旋转
                # img_seg_pad = numpy.clip(numpy.rot90(img_seg_pad, 1), 0, 255).astype(numpy.uint8)  # rotate if necessary

                # cv2.imwrite(patches_save_dir + '@map%s.png' % (
                #         '@' + LT_cur_lon + '@' + LT_cur_lat + '@' + RB_cur_lon + '@' + RB_cur_lat + '@'), img_seg_pad)
                # cv2.imwrite(patches_save_dir + '%s.png' % (
                #         '@' + CT_cur_lon + '@' + CT_cur_lat + '@'), img_seg_pad)

                # print('%s.png' % ('@' + LT_cur_lon + '@' + LT_cur_lat + '@' + RB_cur_lon + '@' + RB_cur_lat + '@'))

                cv2.imwrite(patches_save_dir + f'\\@0@{LT_cur_lon}@{LT_cur_lat}@{RB_cur_lon}@{RB_cur_lat}@{CT_utm_e}@{CT_utm_n}.png', img_seg_pad)

                i += 1

                loc_y = loc_y + stride_y

            loc_x = loc_x + stride_x


if __name__ == '__main__':

    
    basedir = r'E:\QingdaoRawMaps'
    map_dirs = {  
        "2013": r"E:\QingdaoRawMaps\201310\@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.tif",  
        "2017": r"E:\QingdaoRawMaps\201710\@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.tif",
        "2019": r"E:\QingdaoRawMaps\201911\@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.tif",
        "2020": r"E:\QingdaoRawMaps\202002\@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.tif",  
        "2022": r"E:\QingdaoRawMaps\202202\@map@120.42118549346924@36.60643328438966@120.4841423034668@36.573836401969416@.tif"  
    }
    stride_ratios = {  
        "2013": 3,  
        "2017": 4,  
        "2019": 5,  
        "2020": 3,  
        "2022": 4  
    }

    patches_save_root_dir = f"D:\QingdaoMapTiles" + '\\'  
    flight_height = 450  

    total_iterations = len(map_dirs)  # Total iterations  
    current_iteration = 0  # To keep track of progress  
    
    for year, map_dir in map_dirs.items():  
        save_dir_year = os.path.join(patches_save_root_dir, f'{year}_{flight_height}')  
        
        if not os.path.exists(save_dir_year):  
            os.makedirs(save_dir_year)  
        stride_ratio = stride_ratios[year]

        patches_save_dir = save_dir_year  # Save directory for the current year  
        print(f"Saving tiles to: {patches_save_dir} ")  
        
        if not os.path.exists(patches_save_dir):  
            os.mkdir(patches_save_dir)
        
        stride_ratio_str = f'1/{stride_ratio}'

        generate_map_tiles(map_dir, stride_ratio_str, patches_save_dir)
        
        current_iteration += 1  # Increment the progress counter  

        # Calculate and display progress  
        progress = (current_iteration / total_iterations) * 100  
        print(f"[Progress] {progress:.2f}% complete")  



