from glob import glob
import pandas as pd
import utm
import cv2
from tqdm import tqdm
import os

def get_heights_from_qingdao_paths(images_paths: list[str]):

    # images_metadatas = [p.split("@") for p in images_paths]
    # heights = [m[2] for m in images_metadatas]
    # # heights = np.array(heights).astype(np.float64)
    # del images_metadatas

    info_list = [image_path.split('/')[-1].split('@') for image_path in images_paths]
    # heights = np.array([info[-4] for info in info_list]).astype(np.float16)
    # heights = torch.tensor(heights, dtype=torch.float16).unsqueeze(1)
    heights = [float(info[4]) for info in info_list]
    # heights = [float(info[3]) for info in info_list]    # NOTE 给VPR切的图像是filename = f'@{year}@{rotation_angle}@{flight_height}@{CT_utm_e}@{CT_utm_n}@.png'，应该是第三个
    return heights

def get_heights_from_city_paths(images_paths: list[str]):
    info_list = [image_path.split('/')[-1].split('@') for image_path in images_paths]
    heights = [float(info[2]) for info in info_list]
    return heights

def get_heights_preds_from_heights(heights: list[float], heights_pred_choice: list[int]):

    def find_closest_height(height, heights_pred_choice):
        return min(heights_pred_choice, key=lambda x:abs(x-height))
    heights_preds = [find_closest_height(h, heights_pred_choice) for h in heights]
    return heights_preds

root_path = '/root/workspace/ctf53sc7v38s73e0mksg/maps/qd-new/qd_test'
root_path = '/root/workspace/ctf53sc7v38s73e0mksg/maps/qd-new/qd_train'
root_path = '/root/workspace/ctf53sc7v38s73e0mksg/maps/HC-cities-test/ct02'
root_path = '/root/workspace/ctf53sc7v38s73e0mksg/maps/HC-cities-test/ct01'

os.makedirs(root_path, exist_ok=True)

# import os
# import shutil

# # 定义源文件夹和目标文件夹
# src_folder = "/root/workspace/ctf53sc7v38s73e0mksg/maps/qd-new/qd_train1"
# dst_folder = "/root/workspace/ctf53sc7v38s73e0mksg/maps/qd-new/qd_train"

# # 遍历源文件夹中的所有文件
# for filename in os.listdir(src_folder):
#     src_file = os.path.join(src_folder, filename)
#     dst_file = os.path.join(dst_folder, filename)
    
#     # 检查是否是文件
#     if os.path.isfile(src_file):
#         # 复制文件
#         shutil.copy2(src_file, dst_file)
#         print(f"文件 {src_file} 已复制到 {dst_file}")

# src_folder = "/root/workspace/ctf53sc7v38s73e0mksg/maps/qd-new/qd_train2"
# dst_folder = "/root/workspace/ctf53sc7v38s73e0mksg/maps/qd-new/qd_train"

# # 遍历源文件夹中的所有文件
# for filename in os.listdir(src_folder):
#     src_file = os.path.join(src_folder, filename)
#     dst_file = os.path.join(dst_folder, filename)
    
#     # 检查是否是文件
#     if os.path.isfile(src_file):
#         # 复制文件
#         shutil.copy2(src_file, dst_file)
#         print(f"文件 {src_file} 已复制到 {dst_file}")

images_paths = sorted(glob(f"{root_path}/**/*.png", recursive=True))

# heights = get_heights_from_qingdao_paths(images_paths)
heights = get_heights_from_city_paths(images_paths)

heights_pred_choice = list(range(125, 700, 50))

heights_preds = get_heights_preds_from_heights(heights, heights_pred_choice)


def get_utms_from_paths(images_paths):
    info_list = [image_path.split('/')[-1].split('@') for image_path in images_paths]
    # heights = np.array([info[-4] for info in info_list]).astype(np.float16)
    # heights = torch.tensor(heights, dtype=torch.float16).unsqueeze(1)
    lons = [float(info[2]) for info in info_list]
    lats = [float(info[3]) for info in info_list]
    utms = [tuple(utm.from_latlon(lat, lon)[0:2]) for lat, lon in zip(lats, lons)]
    # heights = [float(info[3]) for info in info_list]    # NOTE 给VPR切的图像是filename = f'@{year}@{rotation_angle}@{flight_height}@{CT_utm_e}@{CT_utm_n}@.png'，应该是第三个
    return utms

def get_utms_from_paths_city(images_paths):
    info_list = [image_path.split('/')[-1].split('@') for image_path in images_paths]
    utmes = [float(info[-3]) for info in info_list]
    utmns = [float(info[-2]) for info in info_list]
    utms = [(utme, utmn) for utme, utmn in zip(utmes, utmns)]
    return utms

# image_path,query_height,pred_height,in_threshold,utm_e,utm_n

# utms = get_utms_from_paths(images_paths)
utms = get_utms_from_paths_city(images_paths)

utme = [utm[0] for utm in utms]
utmn = [utm[1] for utm in utms]

data = {
    'image_path': images_paths,
    'query_height': heights,
    'pred_height': heights_preds,
    'in_threshold': [True]* len(images_paths),
    'utm_e': utme,
    'utm_n': utmn,
}

csv_path = '/root/workspace/ctf53sc7v38s73e0mksg/maps/qd-new/qd_test_standard.csv'
csv_path = '/root/workspace/ctf53sc7v38s73e0mksg/maps/qd-new/qd_train_standard.csv'
csv_path = '/root/workspace/ctf53sc7v38s73e0mksg/maps/HC-cities-test/ct02_test_standard.csv'
csv_path = '/root/workspace/ctf53sc7v38s73e0mksg/maps/HC-cities-test/ct01_test_standard.csv'

df = pd.DataFrame(data)
df.to_csv(csv_path, header=True, index=False, encoding='utf-8')

resolution_w = 2048
resolution_h = 1536
# 焦距
focal_length = 1200  # TODO: the intrinsics of the camera

base_height = 125

trans_idx_list = [(0, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]   # 分别对应ct,rt,lt,rb,lb
trans_name_list = ['ct', 'rt', 'lt', 'lb', 'rb']

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


def get_csv_info(csv_path: str):
    df = pd.read_csv(csv_path)
    result = df.to_dict(orient='records')
    return result

def cut_img(images_info, tmp_dir:str):
    cut_bar = tqdm(total=len(images_info))

    for image_info in images_info:

        # 格式image_info = {'image_path': images_paths, 'query_height': query_heights, 'pred_height': pred_class_centers, 'in_threshold': dist <= threshold, 'utm_e', 'utm_n'}
        image_path = image_info['image_path']
        pred_height = image_info['pred_height']
        query_height = image_info['query_height']
        utm_e = image_info['utm_e']
        utm_n = image_info['utm_n']
        origin_meters_h, origin_meters_w = photo_area_meters(query_height)
        zoom_ratio = base_height / pred_height
        move_w = origin_meters_w * (1 - zoom_ratio) / 2
        move_h = origin_meters_h * (1 - zoom_ratio) / 2

        image = cv2.imread(image_path)
        height, width, channels = image.shape
        width = int(width)
        height = int(height)
        
        new_width = round(width * zoom_ratio)
        new_height = round(height * zoom_ratio)

        mid_range = [(height - new_height) // 2, (width - new_width) // 2, (height + new_height) // 2, (width + new_width) // 2]
        trans_w = (width - new_width) // 2
        trans_h = - (height - new_height) // 2

        i = 0

        # for i in range(len(trans_idx_list)):
        trans_idx = trans_idx_list[i]
        cropped_utm_gt = (utm_e + trans_idx[0] * move_w, utm_n + trans_idx[1] * move_h)
        
        h1 = max(mid_range[0] + trans_idx[0] * trans_h, 0)
        h2 = min(mid_range[2] + trans_idx[0] * trans_h, height)
        w1 = max(mid_range[1] + trans_idx[1] * trans_w, 0)
        w2 = min(mid_range[3] + trans_idx[1] * trans_w, width)
        new_img = image[h1:h2, w1:w2]
        new_img_resize = cv2.resize(new_img, (width, height), interpolation = cv2.INTER_LANCZOS4)
        filename = os.path.basename(image_path).replace('.png', '')
        new_filename = f'@{trans_name_list[i]}' + filename + f'@{cropped_utm_gt[0]}@{cropped_utm_gt[1]}@.png'
        save_file_path = os.path.join(tmp_dir, new_filename)
        cv2.imwrite(save_file_path, new_img_resize)
        image_info[trans_name_list[i]] = save_file_path
        image_info[trans_name_list[i] + '_utm'] = cropped_utm_gt
        
        # 切取左上角部分
        pass

        cut_bar.update(1)

    return images_info

images_info = get_csv_info(csv_path)
tmp_dir = './tmp_img/qd_test_cut'
tmp_dir = '/root/workspace/ctvsuas7v38s73eo9qlg/maps/qd_125_years/train_photo'
tmp_dir = '/root/workspace/ctvsuas7v38s73eo9qlg/maps/qd_125_years/test_photo'
tmp_dir = '/root/workspace/ctf53sc7v38s73e0mksg/maps/ct02-test-125_new'
tmp_dir = '/root/workspace/ctf53sc7v38s73e0mksg/maps/ct01-test-125_new'
os.makedirs(tmp_dir, exist_ok=True)
images_info = cut_img(images_info, tmp_dir)
