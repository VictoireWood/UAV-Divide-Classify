import os
import logging
from glob import glob
import cv2
def train_cut(scr_folder, dst_folder):
    # NOTE 输入图像必须是512x512的图像
    def pixels_cut_off_each_side(height):
        baseline = (512 - 358) // 2
        cut_off_pixels = baseline * (height - 300 + 150) // (300 - 150)
        return int(cut_off_pixels)
    def get_new_length(height):
        return 512 - 2 * pixels_cut_off_each_side(height)
    def get_height_from_path(path):
            folder_path = os.path.dirname(path)
            folder_name = os.path.basename(folder_path)
            height = int(folder_name)
            return height
    # def get_heights_from_paths(images_paths: list[str]):
    #     heights = [get_height_from_path(image_path) for image_path in images_paths]    # NOTE /ctf53sc7v38s73e0mksg/maps/SUES-200-512x512/drone_view_512/0003/300/13.jpg
    #     return heights
    def get_new_path(old_path:str, scr_folder, dst_folder):
        new_path = old_path.replace(scr_folder, dst_folder)
        return new_path
    def get_new_image(path):
        raw_image = cv2.imread(path)
        height = get_height_from_path(path)
        cut_off_pixels = pixels_cut_off_each_side(height)
        remain_pixels = get_new_length(height)
        new_image = raw_image[cut_off_pixels:remain_pixels+cut_off_pixels, cut_off_pixels:remain_pixels+cut_off_pixels]
        new_image = cv2.resize(new_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        return new_image
    def create_folder(path):
        folder_path = os.path.dirname(path)
        os.makedirs(folder_path, exist_ok=True)

    logging.info(f"Searching training images in {scr_folder}")
    # images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))   # ORIGION
    # sub_folders = [f'{order:04d}' for order in range(1, 121)]
    sub_folders = [f'{order:04d}' for order in range(121, 201)]
    images_paths = []
    for sub_folder in sub_folders:
        current_images_paths = sorted(glob(f"{scr_folder}/{sub_folder}/**/*.jpg", recursive=True))  # SUES 的图像格式是jpg
        images_paths.extend(current_images_paths)
    for path in images_paths:
        new_path = get_new_path(path, scr_folder, dst_folder)
        new_image = get_new_image(path)
        create_folder(new_path)
        cv2.imwrite(new_path, new_image)
    
train_cut('/root/workspace/ctf53sc7v38s73e0mksg/maps/SUES-200-512x512/drone_view_512', '/root/workspace/ctf53sc7v38s73e0mksg/maps/SUES-200-512x512/drone_view_train_new')
