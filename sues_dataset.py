
import os
import PIL
import torch
import random
import logging
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True

# EDIT
from generate_database import flight_heights

def open_image(path):
    return Image.open(path).convert("RGB")

def get_satellite_image_path(image_path):
    image_folder = os.path.dirname(image_path)  # 显示高度的文件夹路径
    class_folder = os.path.dirname(image_folder)
    assert type(class_folder) == str
    base_folder = os.path.dirname(class_folder)
    base_folder_name = os.path.basename(base_folder)
    satellite_image_folder = class_folder.replace(base_folder_name, 'satellite-view')
    satellite_image_path = os.path.join(satellite_image_folder, '0.png')
    return satellite_image_path


class TestDrone(torch.utils.data.Dataset):
    def __init__(self, test_folder, N=5, image_size=256):
        super().__init__()
        logging.debug(f"Searching test images in {test_folder}")    # NOTE test_folder是SUES-200-512x512的那个文件夹路径

        sub_folders = [f'{order:04d}' for order in range(121, 201)]
        images_paths = []
        for sub_folder in sub_folders:
            current_images_paths = sorted(glob(f"{test_folder}/{sub_folder}/**/*.jpg", recursive=True))
            images_paths.extend(current_images_paths)
        # images_paths = sorted(glob(f"{test_folder}/**/*.png", recursive=True))   # EDIT

        logging.debug(f"Found {len(images_paths)} images")
        class_id_group_id = [TrainDataset.get__class_id__group_id(image_path, N) for image_path in images_paths] # EDIT

        self.images_paths = images_paths
        self.class_id = [id[0] for id in class_id_group_id]  # ORIGION
        self.group_id = [id[1] for id in class_id_group_id]

        self.normalize = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        class_id = self.class_id[index]

        pil_image = open_image(image_path)
        # pil_image = T.functional.resize(pil_image, self.shapes[index])
        image = self.normalize(pil_image)
        if isinstance(image, tuple):
            image = torch.stack(image, dim=0)
        return image, class_id


    def __len__(self):
        return len(self.images_paths)

class TestSatellite(torch.utils.data.Dataset):
    def __init__(self, test_folder, N=5, image_size=256):
        super().__init__()
        logging.debug(f"Searching test images in {test_folder}")

        sub_folders = [f'{order:04d}' for order in range(121, 201)]
        images_paths = []
        for sub_folder in sub_folders:
            current_images_paths = sorted(glob(f"{test_folder}/satellite-view/{sub_folder}/*.png", recursive=True))
            images_paths.extend(current_images_paths)

        logging.debug(f"Found {len(images_paths)} test satellite images")
        class_id_group_id = [self.get__class_id__group_id(path, N) for path in images_paths] # ORIGION

        self.images_paths = images_paths
        self.class_id = [id[0] for id in class_id_group_id]  # ORIGION
        self.group_id = [id[1] for id in class_id_group_id]

        self.normalize = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        class_id = self.class_id[index]

        pil_image = open_image(image_path)
        # pil_image = T.functional.resize(pil_image, self.shapes[index])
        image = self.normalize(pil_image)
        if isinstance(image, tuple):
            image = torch.stack(image, dim=0)
        return image, class_id

    def __len__(self):
        return len(self.images_paths)

    @staticmethod
    def get__class_id__group_id(image_path:str, N):     # EDIT
        """Return class_id and group_id for a given point.
            The class_id is a tuple of UTM_east, UTM_north (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1), and it is between (0, 0) and (N, N).
        # EDIT 针对satellite图像的路径寻找class_id和group_id
        """
        # 获取文件的父文件夹路径
        parent_dir = os.path.dirname(image_path)
        # 获取父文件夹的父文件夹的名称
        parent_name = os.path.basename(parent_dir) # 遥感图像的父文件夹就是地区的编号
        class_id = int(parent_name)
        group_id = class_id % N

        return class_id, group_id

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_path, dataset_name, group_num, N=5, min_images_per_class=10, transform=None):
        """
        Parameters
        ----------
        M : int, the length of the side of each cell in meters.
        N : int, distance (M-wise) between two classes of the same group.
        classes_ids : list of IDs of each class within this group. Each ID is a tuple
                with the center of the cell in UTM coords, e.g: (549900, 4178820).
        images_per_class : dict where the key is a class ID, and the value is a list
                containing the paths of the images withing the class.
        transform : a transform for data augmentation
        """
        super().__init__()

        cache_filename = f"cache/{dataset_name}_N{N}_mipc{min_images_per_class}.torch" # ORIGION

        if not os.path.exists(cache_filename):
            classes_per_group, images_per_class_per_group = initialize(train_path, dataset_name, N, min_images_per_class)
            torch.save((classes_per_group, images_per_class_per_group), cache_filename)
        else:
            classes_per_group, images_per_class_per_group = torch.load(cache_filename)
        classes_ids = classes_per_group[group_num]
        images_per_class = images_per_class_per_group[group_num]

        self.train_path = train_path
        self.N = N
        self.transform = transform
        self.classes_ids = classes_ids
        self.images_per_class = images_per_class
    
    def __getitem__(self, _):
        # The index is ignored, and each class is sampled uniformly
        class_num = random.randint(0, len(self.classes_ids)-1)
        class_id = self.classes_ids[class_num]
        # Pick a random image among the ones in this class.
        image_path = self.train_path + random.choice(self.images_per_class[class_id])
        satellite_image_path = get_satellite_image_path(image_path)
        
        try:
            pil_image = open_image(image_path)
            pil_satellite_image = open_image(satellite_image_path)
            tensor_image = self.transform(pil_image)
            tensor_satellite_image = self.transform(pil_satellite_image)
            image_pair = torch.stack([tensor_image, tensor_satellite_image])
            class_num_pair = torch.tensor(class_num).repeat(2)
        except PIL.UnidentifiedImageError:
            logging.info(f"ERR: There was an error while reading image {image_path}, it is probably corrupted")
            tensor_image = torch.zeros([3, 224, 224])
            image_pair = torch.stack([tensor_image, tensor_image])
            class_num_pair = torch.tensor(class_num).repeat(2)


        return image_pair, class_num_pair

    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])

    def get_classes_num(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)

    def __len__(self):
        """Return a large number. This is because if you return the number of
        classes and it is too small (like in pitts30k), the dataloader within
        InfiniteDataLoader is often recreated (and slows down training).
        """
        return 1500000     # NOTE Found 1459192 images，之前的数额比实际的小，有可能是因为这个才卡住吗？

    @staticmethod
    def get__class_id__group_id(image_path:str, N):     # EDIT
        """
        Return class_id and group_id for a given point.
            The class_id is a tuple of UTM_east, UTM_north (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1), and it is between (0, 0) and (N, N).
        # 通过路径得到高度分组
        """
        # 获取文件的父文件夹路径
        parent_dir = os.path.dirname(image_path)
        # 再次获取父文件夹的父文件夹路径
        grandparent_dir = os.path.dirname(parent_dir)
        # 获取父文件夹的父文件夹的名称
        grandparent_name = os.path.basename(grandparent_dir)
        class_id = int(grandparent_name)
        group_id = class_id % N

        return class_id, group_id



def initialize(dataset_folder, dataset_name, N, min_images_per_class):
    paths_file = f"cache/paths_{dataset_name}.torch"
    # satellite_paths_file = f"cache/satellite_paths_{dataset_name}.torch"
    # Search paths of dataset only the first time, and save them in a cached file
    if not os.path.exists(paths_file):
    # if not os.path.exists(paths_file) or not os.path.exists(satellite_paths_file):
        logging.info(f"Searching training images in {dataset_folder}")
        # images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))   # ANCHOR
        sub_folders = [f'{order:04d}' for order in range(1, 121)]
        images_paths = []
        for sub_folder in sub_folders:
            current_images_paths = sorted(glob(f"{dataset_folder}/{sub_folder}/**/*.jpg", recursive=True))  # SUES 的图像格式是jpg
            images_paths.extend(current_images_paths)
            # current_satellite_images_paths = sorted(glob(f"{dataset_folder}/**/*.png", recursive=True))   # REVIEW
        # Remove folder_path from images_path, so that the same cache file can be used on any machine
        images_paths = [p.replace(dataset_folder, "") for p in images_paths]
        # satellite_images_paths = [get_satellite_image_path(p) for p in images_paths]
        os.makedirs("cache", exist_ok=True)
        torch.save(images_paths, paths_file)
        # torch.save(satellite_images_paths, satellite_paths_file)
    else:
        images_paths = torch.load(paths_file)
        # satellite_images_paths = torch.load(satellite_paths_file)

    logging.info(f"Found {len(images_paths)} images")

    logging.info("For each image, get its UTM east, UTM north from its path")
    logging.info("For each image, get class and group to which it belongs")
    class_id__group_id = [TrainDataset.get__class_id__group_id(path, N) for path in images_paths]

    logging.info("Group together images belonging to the same class")
    images_per_class = defaultdict(list)
    images_per_class_per_group = defaultdict(dict)
    # for image_path, satellite_image_path, (class_id, _) in zip(images_paths, satellite_images_paths, class_id__group_id):
    for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
        images_per_class[class_id].append(image_path)
        

    # Images_per_class is a dict where the key is class_id, and the value
    # is a list with the paths of images within that class.
    images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}

    logging.info("Group together classes belonging to the same group")
    # Classes_per_group is a dict where the key is group_id, and the value
    # is a list with the class_ids belonging to that group.
    classes_per_group = defaultdict(set)
    for class_id, group_id in class_id__group_id:
        if class_id not in images_per_class:
            continue  # Skip classes with too few images
        classes_per_group[group_id].add(class_id)

    for group_id, group_classes in classes_per_group.items():
        for class_id in group_classes:
            images_per_class_per_group[group_id][class_id] = images_per_class[class_id]
    # Convert classes_per_group to a list of lists.
    # Each sublist represents the classes within a group.
    classes_per_group = [list(c) for c in classes_per_group.values()]
    images_per_class_per_group = [c for c in images_per_class_per_group.values()]

    return classes_per_group, images_per_class_per_group



class STrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_path, dataset_name, group_num, N=5, min_images_per_class=10, transform=None):
        """
        Parameters
        ----------
        M : int, the length of the side of each cell in meters.
        N : int, distance (M-wise) between two classes of the same group.
        classes_ids : list of IDs of each class within this group. Each ID is a tuple
                with the center of the cell in UTM coords, e.g: (549900, 4178820).
        images_per_class : dict where the key is a class ID, and the value is a list
                containing the paths of the images withing the class.
        transform : a transform for data augmentation
        """
        super().__init__()

        cache_filename = f"cache/{dataset_name}_N{N}_mipc{min_images_per_class}.torch" # ORIGION

        if not os.path.exists(cache_filename):
            classes_per_group, images_per_class_per_group = initialize(train_path, dataset_name, N, min_images_per_class)
            torch.save((classes_per_group, images_per_class_per_group), cache_filename)
        else:
            classes_per_group, images_per_class_per_group = torch.load(cache_filename)
        classes_ids = classes_per_group[group_num]
        images_per_class = images_per_class_per_group[group_num]

        self.train_path = train_path
        self.N = N
        self.transform = transform
        self.classes_ids = classes_ids
        self.images_per_class = images_per_class
    
    def __getitem__(self, _):
        # The index is ignored, and each class is sampled uniformly
        class_num = random.randint(0, len(self.classes_ids)-1)
        class_id = self.classes_ids[class_num]
        class_center = self.class_centers[class_num]

        # Pick a random image among the ones in this class.
        image_path = self.train_path + random.choice(self.images_per_class[class_id])
        
        try:
            pil_image = open_image(image_path)
            tensor_image = self.transform(pil_image)
        except PIL.UnidentifiedImageError:
            logging.info(f"ERR: There was an error while reading image {image_path}, it is probably corrupted")
            tensor_image = torch.zeros([3, 224, 224])

        return tensor_image, class_num, class_center

    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])

    def get_classes_num(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)

    def __len__(self):
        """Return a large number. This is because if you return the number of
        classes and it is too small (like in pitts30k), the dataloader within
        InfiniteDataLoader is often recreated (and slows down training).
        """
        # return 1000000
        # return 372390
        # return 1459192
        return 1500000     # NOTE Found 1459192 images，之前的数额比实际的小，有可能是因为这个才卡住吗？

    @staticmethod
    # def get__class_id__group_id(utm_east, utm_north, M, N):   # ORIGION
    def get__class_id__group_id(image_path:str, N):     # EDIT
        """Return class_id and group_id for a given point.
            The class_id is a tuple of UTM_east, UTM_north (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1), and it is between (0, 0) and (N, N).
        # EDIT 需要加上高度分组
        """
        # 获取文件的父文件夹路径
        parent_dir = os.path.dirname(image_path)
        # 获取父文件夹的父文件夹的名称
        parent_name = os.path.basename(parent_dir) # 遥感图像的父文件夹就是地区的编号
        class_id = int(parent_name)
        group_id = class_id % N

        return class_id, group_id