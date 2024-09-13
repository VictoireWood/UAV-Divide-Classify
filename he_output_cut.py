import torch
import cv2
import numpy as np
import os
import glob
from tqdm import trange, tqdm
import platform

# import pywt
from PIL import Image


# def dwt_interpolation(img_array, new_size):
    
#     # 读取输入图像
#     # img = Image.open('input_image.jpg')
#     # img_array = np.array(img)

#     # 设置插值尺寸
#     # new_size = (340, 480)

#     wavelet = 'haar'

#     # 将图像分成三个颜色通道
#     r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

#     # 实现小波变换
#     coeffs_r = pywt.wavedec2(r, wavelet, level=2)
#     coeffs_g = pywt.wavedec2(g, wavelet, level=2)
#     coeffs_b = pywt.wavedec2(b, wavelet, level=2)

#     # 插值小波系数
#     coeffs_r[0] = np.interp(np.arange(new_size[0]), np.arange(r.shape[0]), coeffs_r[0])
#     coeffs_r[1] = np.interp(np.arange(new_size[1]), np.arange(r.shape[1]), coeffs_r[1])
#     coeffs_g[0] = np.interp(np.arange(new_size[0]), np.arange(g.shape[0]), coeffs_g[0])
#     coeffs_g[1] = np.interp(np.arange(new_size[1]), np.arange(g.shape[1]), coeffs_g[1])
#     coeffs_b[0] = np.interp(np.arange(new_size[0]), np.arange(b.shape[0]), coeffs_b[0])
#     coeffs_b[1] = np.interp(np.arange(new_size[1]), np.arange(b.shape[1]), coeffs_b[1])

#     # 实现反小波变换
#     reconstructed_r = pywt.waverec2(coeffs_r, wavelet)
#     reconstructed_g = pywt.waverec2(coeffs_g, wavelet)
#     reconstructed_b = pywt.waverec2(coeffs_b, wavelet)

#     # 合并颜色通道
#     reconstructed_img = cv2.merge((reconstructed_r, reconstructed_g, reconstructed_b))

#     # 保存插值图像
#     reconstructed_img = Image.fromarray(reconstructed_img.astype(np.uint8))
#     reconstructed_img.save('interpolated_image.jpg')

#     return reconstructed_img



size = (360, 480)   # cv2.resize里的size是(w, h)的格式，但是imread是(h, w)的格式
resize_size = (size[1], size[0])
base_height = 150   # 最低高度是150米

src_dirs = ['/root/workspace/dcqddb_test/VPR_h630/', '/root/workspace/dcqddb_test/VPR_h400/', '/root/workspace/dcqddb_test/VPR2/']
src_heights = [630, 400, 200]
dst_dir = '/root/workspace/fake_test/'

# tmp_save_path = '/root/workspace/fake_test/VPR_h630/'
# input_path = '/root/workspace/dcqddb_test/VPR_h630/@none@630@727927.2284418452@4068986.901224947@.png'

def cut_img(input_img_path:str, height_estimate, save_dir:str):

    input_path = input_img_path
    tmp_save_path = save_dir

    filename = os.path.basename(input_path)
    filename_without_extension = os.path.splitext(filename)[0]


    origin_input = cv2.imread(input_path)
    h_in, w_in = origin_input.shape[:2]



    # height_estimate = 630
    # NOTE 这里其实应该用前面模型求出的高度做推断inference

    if height_estimate <= base_height:
        height_estimate = base_height
    cut_ratio = base_height / height_estimate

    origin_img_cut_h_pixel = round(h_in * cut_ratio)
    origin_img_cut_w_pixel = round(w_in * cut_ratio)


    ct_cut = [round((h_in - origin_img_cut_h_pixel) / 2), round((h_in + origin_img_cut_h_pixel) / 2), round((w_in - origin_img_cut_w_pixel) / 2), round((w_in + origin_img_cut_w_pixel) / 2)]
    lt_cut = [0, round(origin_img_cut_h_pixel), 0, round(origin_img_cut_w_pixel)]
    lb_cut = [0, round(origin_img_cut_h_pixel), round(w_in - origin_img_cut_w_pixel), w_in]
    rt_cut = [round(h_in - origin_img_cut_h_pixel), h_in, 0, round(origin_img_cut_w_pixel)]
    rb_cut = [round(h_in - origin_img_cut_h_pixel), h_in, round(w_in - origin_img_cut_w_pixel), w_in]

    ct_origin = origin_input[ct_cut[0]:ct_cut[1], ct_cut[2]:ct_cut[3]]
    lt_origin = origin_input[lt_cut[0]:lt_cut[1], lt_cut[2]:lt_cut[3]]
    lb_origin = origin_input[lb_cut[0]:lb_cut[1], lb_cut[2]:lb_cut[3]]
    rt_origin = origin_input[rt_cut[0]:rt_cut[1], rt_cut[2]:rt_cut[3]]
    rb_origin = origin_input[rb_cut[0]:rb_cut[1], rb_cut[2]:rb_cut[3]]

    # ct_resize = cv2.resize(ct_origin, size)
    ct_resize = cv2.resize(ct_origin, resize_size)
    lt_resize = cv2.resize(lt_origin, resize_size)
    lb_resize = cv2.resize(lb_origin, resize_size)
    rt_resize = cv2.resize(rt_origin, resize_size)
    rb_resize = cv2.resize(rb_origin, resize_size)

    if not os.path.exists(tmp_save_path):
        os.makedirs(tmp_save_path)

    if platform.system() == "Windows":
        slash = '\\'
    else:
        slash = '/'

    cv2.imwrite(tmp_save_path + f'{slash}@ct{filename_without_extension}.png', ct_resize)
    cv2.imwrite(tmp_save_path + f'{slash}@lt{filename_without_extension}.png', lt_resize)
    cv2.imwrite(tmp_save_path + f'{slash}@lb{filename_without_extension}.png', lb_resize)
    cv2.imwrite(tmp_save_path + f'{slash}@rt{filename_without_extension}.png', rt_resize)
    cv2.imwrite(tmp_save_path + f'{slash}@rb{filename_without_extension}.png', rb_resize)

def get_png_files_paths(directory):
    # 使用 glob 模块查找所有 .png 文件
    png_files = glob.glob(os.path.join(directory, "*.png"))
    
    # 将相对路径转换为绝对路径
    # absolute_paths = [os.path.abspath(file) for file in png_files]
    
    return png_files

for idx in range(len(src_dirs)):
    dir = src_dirs[idx]
    height = src_heights[idx]
    try:
        dir = os.path.dirname(dir)
    except:
        pass
    base_dir = os.path.basename(dir)
    dst_child_dir = os.path.join(dst_dir, base_dir)
    print(f'\n{base_dir}:\n')

    if not os.path.exists(dst_child_dir):
        os.makedirs(dst_child_dir)

    png_files_paths = get_png_files_paths(dir)
    file_num = len(png_files_paths)

    tqdm_bar = tqdm(range(file_num), ncols=100, desc="")
    
    num=0
    for png_file_path in png_files_paths:
        cut_img(png_file_path, height, dst_child_dir)
        num += 1
        tqdm_bar.set_postfix(rate=num/file_num,tiles=num)
        _ = tqdm_bar.update()
        # _ = tqdm_bar.refresh()
