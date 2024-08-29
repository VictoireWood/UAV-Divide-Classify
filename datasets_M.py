
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
from generate_database import flight_heights, M_list

h_num = len(flight_heights)
h_start = flight_heights[0] - (flight_heights[1] - flight_heights[0])/2
h_end = flight_heights[-1] + (flight_heights[-1] - flight_heights[-2])/2
mid_heights = [h_start]
for i in range(h_num - 1):
    mid_h = flight_heights[i] + (flight_heights[i+1] - flight_heights[i])/2
    mid_heights.append(mid_h)
mid_heights.append(h_end)

# def h2M(h):

def open_image(path):
    return Image.open(path).convert("RGB")


def get_h_utmeast_utmnorth(images_paths):
    images_metadatas = [p.split("@") for p in images_paths]
    # field 1 is UTM east, field 2 is UTM north
    # self.utmeast_utmnorth = np.array([(m[1], m[2]) for m in images_metadatas]).astype(np.float64)   # ANCHOR: 原始
    # self.utmeast_utmnorth = np.array([(m[-3], m[-2]) for m in images_metadatas]).astype(np.float64)   # REVIEW: 邵星雨改，我设置的图片格式是@角度（默认0）@UTM-east@UTM-north@.png
    h_utmeast_utmnorth = np.array([(m[-4], m[-3], m[-2]) for m in images_metadatas]).astype(np.float64)   # EDIT: 邵星雨改，我设置的图片格式是@角度（默认0）@高度@UTM-east@UTM-north@.png
    return h_utmeast_utmnorth

class TestDataset(torch.utils.data.Dataset):
    # def __init__(self, test_folder, M=10, N=5, image_size=256):   # ORIGION
    def __init__(self, test_folder, N=5, image_size=256): # EDIT
        super().__init__()
        logging.debug(f"Searching test images in {test_folder}")

        # images_paths = sorted(glob(f"{test_folder}/**/*.jpg", recursive=True))    # ANCHOR: 原始
        images_paths = sorted(glob(f"{test_folder}/**/*.png", recursive=True)) # REVIEW: 邵星雨改

        logging.debug(f"Found {len(images_paths)} images")
        
        '''# EDIT 这里写成函数方便复用
        images_metadatas = [p.split("@") for p in images_paths]
        # field 1 is UTM east, field 2 is UTM north
        # self.utmeast_utmnorth = np.array([(m[1], m[2]) for m in images_metadatas]).astype(np.float64)   # ANCHOR: 原始
        # self.utmeast_utmnorth = np.array([(m[-3], m[-2]) for m in images_metadatas]).astype(np.float64)   # REVIEW: 邵星雨改，我设置的图片格式是@角度（默认0）@UTM-east@UTM-north@.png
        self.h_utmeast_utmnorth = np.array([(m[-4], m[-3], m[-2]) for m in images_metadatas]).astype(np.float64)   # EDIT: 邵星雨改，我设置的图片格式是@角度（默认0）@高度@UTM-east@UTM-north@.png'''
        self.h_utmeast_utmnorth = get_h_utmeast_utmnorth(images_paths)

        # class_id_group_id = [TrainDataset.get__class_id__group_id(*m, M, N) for m in self.utmeast_utmnorth] # ORIGION
        # class_id_group_id = [TrainDataset.get__class_id__group_id(*m, M, N, train_dataset=False) for m in self.h_utmeast_utmnorth] # EDIT 1
        class_id_group_id_M = [TrainDataset.get__class_id__group_id__M(*m, N, train_dataset=False) for m in self.h_utmeast_utmnorth]

        self.images_paths = images_paths
        # self.class_id = [(id[0][0]+ M // 2, id[0][1]+ M // 2) for id in class_id_group_id]  # ORIGION 求中心点UTM
        # self.group_id = [id[1] for id in class_id_group_id] # ORIGION 求group

        self.class_id = [(id_M[0][0], id_M[0][1]+ id_M[-1] // 2, id_M[0][2]+ id_M[-1] // 2) for id_M in class_id_group_id_M] # EDIT 已经包含了M的信息，class_id是(h, utm_e, utm_n)，应该取第二个第三个求坐标
        self.group_id = [id_M[1] for id_M in class_id_group_id_M]   # NOTE class_id, group_id, M

        

        self.normalize = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        # class_id = self.class_id[index]

        pil_image = open_image(image_path)
        # pil_image = T.functional.resize(pil_image, self.shapes[index])
        image = self.normalize(pil_image)
        if isinstance(image, tuple):
            image = torch.stack(image, dim=0)
        # return image, tuple(self.utmeast_utmnorth[index]) # ORIGION 这里忘改成h_utmeast_utmnorth了
        return image, tuple(self.h_utmeast_utmnorth[index]) # EDIT 返回类别


    def __len__(self):
        return len(self.images_paths)

    def get_classes_num(self):
        return len(self.dict__cell_id__class_num)



class TrainDataset(torch.utils.data.Dataset):
    # def __init__(self, train_path, dataset_name, group_num, M=10, N=5, min_images_per_class=10, transform=None):    # ORIGION
    def __init__(self, train_path, dataset_name, group_num, N=5, min_images_per_class=10, transform=None):    # EDIT 不输入M，自适应高度
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

        # cache_filename = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_images_per_class}.torch" # ORIGION
        cache_filename = f"cache/{dataset_name}_N{N}_H{flight_heights[0]}-{flight_heights[-1]}_mipc{min_images_per_class}.torch" # EDIT 如果M改为自适应的话M会一直变

        if not os.path.exists(cache_filename):
            # classes_per_group, images_per_class_per_group = initialize(train_path, dataset_name, M, N, min_images_per_class)    # ORIGION
            classes_per_group, images_per_class_per_group = initialize(train_path, dataset_name, N, min_images_per_class)    # EDIT
            torch.save((classes_per_group, images_per_class_per_group), cache_filename)
        else:
            classes_per_group, images_per_class_per_group = torch.load(cache_filename)
        classes_ids = classes_per_group[group_num]
        images_per_class = images_per_class_per_group[group_num]

        self.train_path = train_path
        flight_height = classes_ids[0][0]
        M_index = flight_heights.index(flight_height)
        M = M_list[M_index]
        self.M = M
        self.N = N
        self.transform = transform
        self.classes_ids = classes_ids
        self.images_per_class = images_per_class
        # self.class_centers = [(cl_id[0] + M // 2, cl_id[1] + M // 2) for cl_id in self.classes_ids] # ORIGION 这里id只有俩，得加上h
        self.class_centers = [(cl_id[0], cl_id[1] + M // 2, cl_id[2] + M // 2) for cl_id in self.classes_ids]# EDIT
    
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
        return 1000000

    @staticmethod
    # def get__class_id__group_id(utm_east, utm_north, M, N):   # ORIGION
    # def get__class_id__group_id(h, utm_east, utm_north, M, N, train_dataset = True):     # EDIT 1
    def get__class_id__group_id__M(h, utm_east, utm_north, N, train_dataset = True):     # EDIT 2
        """Return class_id and group_id for a given point.
            The class_id is a tuple of UTM_east, UTM_north (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1), and it is between (0, 0) and (N, N).
        # EDIT 需要加上高度分组
        """
        # EDIT
        # 把这里判断的改成函数？
        # mid_h_num = len(flight_heights) - 1
        # h_num = len(flight_heights)
        # h_start = flight_heights[0] - (flight_heights[1] - flight_heights[0])/2
        # h_end = flight_heights[-1] + (flight_heights[-1] - flight_heights[-2])/2
        # mid_heights = [h_start]
        # for i in range(mid_h_num):
        #     mid_h = flight_heights[i] + (flight_heights[i+1] - flight_heights[i])/2
        #     mid_heights.append(mid_h)
        # mid_heights.append(h_end)

        h_group_id = 0
        for i in range(h_num):
            if h > mid_heights[i] and h < mid_heights[i+1]:
                h_group_id = i + 1  # 从1开始
                h_class_id = flight_heights[i]  # NOTE: class_id设置成所在区间“中间”高度，也就是切的时候的基准高度
                M = M_list[i]   # NOTE: 自适应M
                break
        if h_group_id == 0: # NOTE: 这里的0相当于是SALAD里的dustbin，class_id将0作为dustbin
            if train_dataset:   # 如果是训练集
                logging.debug(f"Found a image's flight height cannot be classified: @*@{h}@{utm_east}@{utm_north}@.png, h_group_id, h_class_id = 0")
                h_class_id = 0  # NOTE: class_id设成零，因为无人机不可能在地面
                M = None
            else:               # 如果是测试集
                if h < mid_heights[0]:
                    h_group_id, h_class_id, M = 1, flight_heights[0], M_list[1]
                else:
                    h_group_id, h_class_id, M = h_num, flight_heights[-1], M_list[-1]
        # 这里的M需要向函数外输出
        # !EDIT

        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)

        # class_id = (rounded_utm_east, rounded_utm_north)    # ORIGION
        class_id = (h_class_id, rounded_utm_east, rounded_utm_north)    # EDIT
        
        # group_id goes from (0, 0) to (N, N)
        '''# ORIGION
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M)'''
        # EDIT
        group_id = (h_group_id,
                    rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M)
        return class_id, group_id, M


# def initialize(dataset_folder, dataset_name, M, N, min_images_per_class):   # ORIGION 需要传出M
def initialize(dataset_folder, dataset_name, N, min_images_per_class):   # 需要传出M
    paths_file = f"cache/paths_{dataset_name}.torch"
    # Search paths of dataset only the first time, and save them in a cached file
    if not os.path.exists(paths_file):
        logging.info(f"Searching training images in {dataset_folder}")
        # images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))   # ANCHOR
        images_paths = sorted(glob(f"{dataset_folder}/**/*.png", recursive=True))   # REVIEW
        # Remove folder_path from images_path, so that the same cache file can be used on any machine
        images_paths = [p.replace(dataset_folder, "") for p in images_paths]
        os.makedirs("cache", exist_ok=True)
        torch.save(images_paths, paths_file)
    else:
        images_paths = torch.load(paths_file)

    logging.info(f"Found {len(images_paths)} images")

    '''# EDIT 函数复用
    images_metadatas = [p.split("@") for p in images_paths]
    # field 1 is UTM east, field 2 is UTM north
    # utmeast_utmnorth = [(m[1], m[2]) for m in images_metadatas] # ANCHOR
    # utmeast_utmnorth = [(m[-3], m[-2]) for m in images_metadatas]   # EDIT version 1
    h_utmeast_utmnorth = [(m[-4], m[-3], m[-2]) for m in images_metadatas]  # EDIT: version 2，我设置的图片格式是@角度（默认0）@高度@UTM-east@UTM-north@.png
    # utmeast_utmnorth = np.array(utmeast_utmnorth).astype(np.float64)    # ORIGION
    h_utmeast_utmnorth = np.array(h_utmeast_utmnorth).astype(np.float64)    # EDIT 这个分割过程的另一次出现是在TrainDataset的初始化过程
    del images_metadatas'''

    h_utmeast_utmnorth = get_h_utmeast_utmnorth(images_paths)

    # 判断h在哪个高度区间，根据区间进行M的设置
    

    logging.info("For each image, get its UTM east, UTM north from its path")
    logging.info("For each image, get class and group to which it belongs")
    # class_id__group_id = [TrainDataset.get__class_id__group_id(*m, M, N) for m in utmeast_utmnorth] # ORIGION
    # class_id__group_id = [TrainDataset.get__class_id__group_id(*m, M, N, train_dataset=True) for m in h_utmeast_utmnorth]   # EDIT 1
    class_id__group_id__M = [TrainDataset.get__class_id__group_id__M(*m, N, train_dataset=False) for m in h_utmeast_utmnorth]    # EDIT 2 需要输出自适应M
    # class_id__group_id = [c[0:1] for c in class_id__group_id__M]
    # M_per_class = [c[-1] for c in class_id__group_id__M]



    logging.info("Group together images belonging to the same class")
    images_per_class = defaultdict(list)
    images_per_class_per_group = defaultdict(dict)

    # for image_path, (class_id, _, M) in zip(images_paths, class_id__group_id): # ANCHOR
    #     images_per_class[class_id].append(image_path)

    for image_path, (class_id, _, _) in zip(images_paths, class_id__group_id__M): # EDIT
        images_per_class[class_id].append(image_path)

    # Images_per_class is a dict where the key is class_id, and the value
    # is a list with the paths of images within that class.
    images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}

    logging.info("Group together classes belonging to the same group")
    # Classes_per_group is a dict where the key is group_id, and the value
    # is a list with the class_ids belonging to that group.
    classes_per_group = defaultdict(set)

    '''for class_id, group_id in class_id__group_id:   # ORIGION
        if class_id not in images_per_class:
            continue  # Skip classes with too few images
        classes_per_group[group_id].add(class_id)'''
    
    for class_id, group_id, _ in class_id__group_id__M:   # EDIT
        if class_id not in images_per_class:
            continue  # Skip classes with too few images
        classes_per_group[group_id].add(class_id)       # FIXME 这里调试一下datasets看看这里应该怎么加

    for group_id, group_classes in classes_per_group.items():
        for class_id in group_classes:
            images_per_class_per_group[group_id][class_id] = images_per_class[class_id]
    # Convert classes_per_group to a list of lists.
    # Each sublist represents the classes within a group.
    classes_per_group = [list(c) for c in classes_per_group.values()]
    images_per_class_per_group = [c for c in images_per_class_per_group.values()]

    return classes_per_group, images_per_class_per_group

