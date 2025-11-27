
import torch
from tqdm import tqdm

from classifiers import LinearLayer

import numpy as np

import torchvision.transforms as T

from sklearn import linear_model

import faiss

from he_datasets import RetrievalDataset

# EDIT
import parser
args = parser.parse_arguments()



LR_N = [1, 5, 10, 20]
# threshold = 25  # ORIGION 原来的M=20m，这里设置的是偏移25米算成功召回
# threshold = 500 # EDIT 这里是要配合M的设定，M设为800米
threshold = args.threshold  # EDIT
if threshold is None:
    threshold_M_ratio = 1
    threshold = args.M * threshold_M_ratio

def compute_pred(criterion, descriptors):
    if isinstance(criterion, LinearLayer):
        # Using LinearLayer
        return criterion(descriptors, None)[0]
    else:
        # Using AMCC/LMCC
        # return torch.mm(descriptors, criterion.weight.t())
        try:
            p = torch.mm(descriptors, criterion.weight.t())
            return p
        except:
            p = torch.mm(descriptors, criterion.weight)
            return p    # 返回的是描述子和各个class原型向量的相似度


#### Validation
def inference(args, model, classifiers, test_dl, groups, num_test_images):
    
    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]
    valid_distances = torch.zeros(num_test_images, max(LR_N))
        
    all_preds_utm_centers = [center for group in groups for center in group.class_centers]
    all_preds_utm_centers = torch.tensor(all_preds_utm_centers).to(args.device)
    
    with torch.no_grad():
        # query_class are the UTMs of the center of the class to which the query belongs
        for query_i, (images, query_utms) in enumerate(tqdm(test_dl, ncols=100)):
            images = images.to(args.device)
            query_utms = torch.tensor(query_utms).to(args.device)
            descriptors = model(images)
            
            all_preds_confidences = torch.zeros([0], device=args.device)
            for i in range(len(classifiers)):
                pred = compute_pred(classifiers[i], descriptors)
                assert pred.shape[0] == 1  # pred has shape==[1, num_classes]
                all_preds_confidences = torch.cat([all_preds_confidences, pred[0]])
                
            topn_pred_class_id = all_preds_confidences.argsort(descending=True)[:max(LR_N)]
            pred_class_id = all_preds_utm_centers[topn_pred_class_id]
            dist=torch.cdist(query_utms.unsqueeze(0), pred_class_id.to(torch.float64))
            valid_distances[query_i] = dist
                
    classifiers = [c.cpu() for c in classifiers]
    torch.cuda.empty_cache()  # Release classifiers memory
    lr_ns = []
    for N in LR_N:
        lr_ns.append(torch.count_nonzero((valid_distances[:, :N] <= threshold).any(axis=1)).item() * 100 / num_test_images)

    gcd_str = ", ".join([f'LR@{N}: {acc:.2f}' for N, acc in zip(LR_N, lr_ns)])
    
    return gcd_str


# inference 推理剪切下来5个一组的图像

def inference_he_output(args, model, classifiers, test_dl, groups, images_info:list[dict]):

    def process_dict(input_dict):
        result_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, list):
                if len(value) == 1 and isinstance(value[0], str):
                    result_dict[key] = value[0]
                elif all(isinstance(v, torch.Tensor) for v in value):
                    elements = []
                    for tensor in value:
                        if tensor.numel() == 1:
                            elements.append(tensor.item())
                        else:
                            elements.append(tuple(tensor.tolist()))
                    result_dict[key] = tuple(elements)
                else:
                    result_dict[key] = value
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    result_dict[key] = value.item()
                else:
                    result_dict[key] = tuple(value.tolist())
            else:
                result_dict[key] = value
        return result_dict
    
    num_test_groups = len(images_info)
    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]
    valid_distances = torch.zeros(num_test_groups, max(LR_N))
    pred_top1_groups = torch.zeros(num_test_groups, 5, 2)
    pred_top1_class_ids = torch.zeros(num_test_groups, 5, 2)
        
    all_preds_utm_centers = [center for group in groups for center in group.class_centers]
    all_preds_utm_centers = torch.tensor(all_preds_utm_centers).to(args.device)

    images_info_update = []
    
    with torch.no_grad():
        # query_class are the UTMs of the center of the class to which the query belongs
        for query_i, (images_list, images_info) in enumerate(tqdm(test_dl, ncols=100)):
            images_list = images_list.squeeze(0)    # 否则images_list的大小为torch.Size([1, 5, 3, 336, 448])
            images_list = images_list.to(args.device)
            images_info = process_dict(images_info)
            query_utm = torch.tensor((images_info['utm_e'], images_info['utm_n'])).to(args.device) # 这个是中心点的utm
            pred_group_utms = []
            for i in range(len(images_list)):
                image = images_list[i].unsqueeze(0).to(args.device)
                descripter = model(image)
                all_preds_confidences = torch.zeros([0], device=args.device)
                for j in range(len(classifiers)):
                    pred = compute_pred(classifiers[j], descripter)
                    assert pred.shape[0] == 1  # pred has shape==[1, num_classes]
                    all_preds_confidences = torch.cat([all_preds_confidences, pred[0]])
                topn_pred_class_id = all_preds_confidences.argsort(descending=True)[:max(LR_N)]
                pred_class_id = all_preds_utm_centers[topn_pred_class_id]
                pred_group_utms.append(pred_class_id)
                pred_top1_groups[query_i, i] = pred_group_utms[i][0]    # 记录5个位置的图像预测的top1结果，每次记录5个utm，其实是中心坐标
                # images_info[i] = pred_class_id

                images_info[i] = tuple(pred_group_utms[i][0].tolist())


                all_preds_class_ids = [id for group in groups for id in group.classes_ids]
                all_preds_class_ids = torch.tensor(all_preds_class_ids).to(args.device)
                pred_class_id_real = all_preds_class_ids[topn_pred_class_id]
                top1_pred_class_id_real = pred_class_id_real[0]
                pred_top1_class_ids[query_i, i] = top1_pred_class_id_real

                # images_per_class = {}
                # images_per_class.update(group.images_per_class for group in groups)
                

                
                
            dist=torch.cdist(query_utm.unsqueeze(0).to(torch.float64), pred_group_utms[0].to(torch.float64)).squeeze()  # NOTE 这里没有经过筛选，直接用了中心位置的utm作为估计图像的UTM
            valid_distances[query_i] = dist
            images_info_update.append(images_info)

        



    classifiers = [c.cpu() for c in classifiers]
    torch.cuda.empty_cache()  # Release classifiers memory
    lr_ns = []
    for N in LR_N:
        lr_ns.append(torch.count_nonzero((valid_distances[:, :N] <= threshold).any(axis=1)).item() * 100 / num_test_groups)

    gcd_str = ", ".join([f'LR@{N}: {acc:.2f}' for N, acc in zip(LR_N, lr_ns)])

    return gcd_str, images_info_update

def ransac_fit_line(diagonal_points: np.ndarray, sample_weight: np.ndarray|None):
    assert diagonal_points.shape == (30, 2)
    x = diagonal_points[:, 0]
    x = x.reshape(-1, 1)
    y = diagonal_points[:, 1]

    # 将3个结果的候选点共30个，用RANSAC法拟合直线
    ransac = linear_model.RANSACRegressor()
    # 使用数据拟合直线
    ransac.fit(x, y, sample_weight=sample_weight)
    # if sample_weight is None:
    #     ransac.fit(x, y)
    # else:
    #     ransac.fit(x, y, sample_weight=sample_weight)

    # 获取拟合直线的斜率（系数）和截距
    line_coef = ransac.estimator_.coef_[0]

    # if np.abs(slope) > 10**10:
    #     inlier_mask = ransac.inlier_mask_
    #     x_values_of_inliers = X[inlier_mask]
    #     std_x_inliers = np.std(x_values_of_inliers)
    #     if std_x_inliers < 1e-6:
    #         x_constant = np.mean(x_values_of_inliers)
    #         print("直线方程为 x =", x_constant)

    line_intercept = ransac.estimator_.intercept_

    line_param = np.hstack((line_coef, line_intercept))
    
    return line_param

    # # 生成用于绘制拟合直线的x值范围
    # x_fit = np.linspace(0, 1, 100).reshape(-1, 1)
    # y_fit = line_coef * x_fit + line_intercept

def get_intersection_point(coef_intercept_1, coef_intercept_2):
    if np.isclose(coef_intercept_1[0], coef_intercept_2[0]):
        raise ValueError("两条直线平行，不存在交点")        
    else:
        # 计算交点的x坐标
        # x_intersect = (b2 - b1) / (m1 - m2)
        x_intersect = (coef_intercept_2[1] - coef_intercept_1[1]) / (coef_intercept_1[0] - coef_intercept_2[0])
        # 计算交点的y坐标，代入第一条直线方程计算（也可以代入第二条直线方程）
        # y_intersect = m1 * x_intersect + b1
        y_intersect = coef_intercept_1[0] * x_intersect + coef_intercept_1[1]

        intersection_point = np.hstack((x_intersect, y_intersect))

        return intersection_point

def dist(p1, p2):
        import math
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def is_inside_circle(circle, point):
    center, radius = (circle[0], circle[1]), circle[2]
    return dist(center, point) <= radius

def make_circumcircle(p1, p2, p3):
    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    ux = ((p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p3[1]) + (p2[0] ** 2 + p2[1] ** 2) * (p3[1] - p1[1]) + (p3[0] ** 2 + p3[1] ** 2) * (p1[1] - p2[1])) / d
    uy = ((p1[0] ** 2 + p1[1] ** 2) * (p3[0] - p2[0]) + (p2[0] ** 2 + p2[1] ** 2) * (p1[0] - p3[0]) + (p3[0] ** 2 + p3[1] ** 2) * (p2[0] - p1[0])) / d
    r = dist((ux, uy), p1)
    return (ux, uy, r)

def welzl(points):
    shuffled = list(points)
    circle = (0, 0, 0)
    for i, p in enumerate(shuffled):
        if is_inside_circle(circle, p):
            continue
        circle = (p[0], p[1], 0)
        for j, q in enumerate(shuffled[:i]):
            if is_inside_circle(circle, q):
                continue
            circle = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, dist(p, q) / 2)
            for k, r in enumerate(shuffled[:j]):
                if is_inside_circle(circle, r):
                    continue
                circle = make_circumcircle(p, q, r)
    return circle

def get_cluster_center(one_patch_utms:np.ndarray, sample_weight:np.ndarray|None=None):
    """
    Calculate the center of a cluster of UTM coordinates using One-Class SVM.

    This function identifies inliers from the given UTM coordinates using a
    One-Class SVM model with an RBF kernel. It then calculates the center of
    these inliers. If sample weights are provided, a weighted mean is used to 
    determine the center; otherwise, a simple mean is used.

    Parameters
    ----------
    one_patch_utms : np.ndarray
        A 2D array where each row represents a UTM coordinate (x, y).
    sample_weight : np.ndarray or None, optional
        An array of weights corresponding to the UTM coordinates, by default None.

    Returns
    -------
    np.ndarray
        A 1D array representing the center (x, y) of the inlier UTM coordinates.
    """

    assert one_patch_utms.shape[1] == 2
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(one_patch_utms)
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(n_estimators=1, contamination=0.4)
    model = OneClassSVM(kernel='rbf', nu=0.4).fit(points_scaled)
    predictions = model.predict(points_scaled)
    inliers_indices = np.where(predictions == 1)[0]
    inliers = one_patch_utms[inliers_indices]
    if sample_weight is None:
        center = np.mean(inliers, axis=0)
    else:
        inliers_weight = sample_weight[inliers_indices]
        weighted_sum_x = np.sum(inliers[:, 0] * inliers_weight)
        weighted_sum_y = np.sum(inliers[:, 1] * inliers_weight)
        total_weight = np.sum(inliers_weight)
        center = np.array([weighted_sum_x / total_weight, weighted_sum_y / total_weight])
    return center

def get_inlier(patches_utms:np.ndarray, nu=0.2, threshold=25):
    """
    Get the inlier of a patch of utms.

    Parameters
    ----------
    patches_utms : np.ndarray
        A 2D array of shape (n, 2) representing the UTM coordinates of a patch.
    nu : float, optional
        The parameter of the OneClassSVM model, by default 0.2.
    threshold : int, optional
        The threshold of the radius of the circumcircle, by default 25.

    Returns
    -------
    tuple or None
        (ux, uy) representing the inlier of the patch, or None if the radius of the circumcircle is larger than the threshold.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(patches_utms)
    from sklearn.svm import OneClassSVM
    model = OneClassSVM(kernel='rbf', nu=nu).fit(points_scaled)
    predictions = model.predict(points_scaled)
    inliers_indices = np.where(predictions == 1)[0]
    inliers = patches_utms[inliers_indices]
    
    inliers_list = inliers.tolist()
    inliers_list_without_duplicate = []
    for point in inliers_list:
        tuple_point = tuple(point)
        if tuple_point not in inliers_list_without_duplicate:
            inliers_list_without_duplicate.append(tuple_point)
    ux, uy, r = welzl(inliers_list_without_duplicate)  # (ux,uy,r)
    if r > threshold:
        return (ux, uy)
    else:
        return None

def ransac_mid_point(patches_utms: np.ndarray, sample_weight: np.ndarray|None=None):
    assert patches_utms.shape == (5, 10, 2)
    lt_rb_index = [0, 1, 3]
    rt_lb_index = [0, 2, 4]
    lt_rb = patches_utms[lt_rb_index, :, :]        # 013
    rt_lb = patches_utms[rt_lb_index, :, :]        # 024
    if sample_weight is not None:
        lt_rb_weight = sample_weight[lt_rb_index, :]
        rt_lb_weight = sample_weight[rt_lb_index, :]
        lt_rb_weight = lt_rb_weight.reshape(-1)
        rt_lb_weight = rt_lb_weight.reshape(-1)
    else:
        lt_rb_weight = None
        rt_lb_weight = None

    lt_rb = lt_rb.reshape(-1, 2)
    rt_lb = rt_lb.reshape(-1, 2)

    if np.all(lt_rb == rt_lb):
        mid_patch_utm = patches_utms[0]
        if sample_weight is not None:
            sample_weight_for_mid_patch = sample_weight[0]
        else:
            sample_weight_for_mid_patch = None
        mid_point = get_cluster_center(mid_patch_utm, sample_weight = sample_weight_for_mid_patch)
    else:
        patches_utms = patches_utms.reshape(-1,2)
        circle_mid_point = get_inlier(patches_utms)
        
        if circle_mid_point is None:
            lt_rb_line = ransac_fit_line(lt_rb, lt_rb_weight)
            rt_lb_line = ransac_fit_line(rt_lb, rt_lb_weight)
            mid_point = get_intersection_point(lt_rb_line, rt_lb_line)
        else:
            mid_point = np.array(circle_mid_point)
    return mid_point


def inference_he_output_ransac(args, model, classifiers, test_dl, groups, images_info:list[dict]):

    def process_dict(input_dict):
        result_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, list):
                if len(value) == 1 and isinstance(value[0], str):
                    result_dict[key] = value[0]
                elif all(isinstance(v, torch.Tensor) for v in value):
                    elements = []
                    for tensor in value:
                        if tensor.numel() == 1:
                            elements.append(tensor.item())
                        else:
                            elements.append(tuple(tensor.tolist()))
                    result_dict[key] = tuple(elements)
                else:
                    result_dict[key] = value
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    result_dict[key] = value.item()
                else:
                    result_dict[key] = tuple(value.tolist())
            else:
                result_dict[key] = value
        return result_dict
    
    def mean_down_threshold(distance, retrieval_threshold):

        # 筛选出所有小于retrieval_threshold的元素
        filtered_elements = distance[distance < retrieval_threshold]

        # 计算这些元素的平均值
        if filtered_elements.size > 0:
            average_value = np.mean(filtered_elements)
        
            return average_value

        else:
            average_value = 0
            return average_value
            # raise ValueError("No elements found below the threshold.")

    
    num_test_groups = len(images_info)
    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]
    valid_distances = torch.zeros(num_test_groups, max(LR_N))
    ransac_distances = np.zeros(num_test_groups)
    ransac_with_weight_distances = np.zeros(num_test_groups)
    mid_patch_top1_distances = np.zeros(num_test_groups)
    mid_patch_with_weight_distances = np.zeros(num_test_groups)

    # choose_class_number = 20
    # pred_top1_groups = torch.zeros(num_test_groups, choose_class_number, 2)
    # pred_top1_class_ids = torch.zeros(num_test_groups, choose_class_number, 2)
    # pred_top3_class_ids = torch.zeros(num_test_groups, choose_class_number, 3, 2)
    # pred_top5_class_ids = torch.zeros(num_test_groups, choose_class_number, 5, 2).type(torch.int64)
    

    pred_top1_groups = torch.zeros(num_test_groups, 5, 2)
    pred_top1_class_ids = torch.zeros(num_test_groups, 5, 2)
    pred_top3_class_ids = torch.zeros(num_test_groups, 5, 3, 2)
    pred_top5_class_ids = torch.zeros(num_test_groups, 5, 5, 2).type(torch.int64)
    pred_top10_class_ids = torch.zeros(num_test_groups, 5, 10, 2).type(torch.int64)
    
    ransac_pred_utms = np.zeros((num_test_groups, 2), dtype=np.float64)
    ransac_pred_utms_with_weight = np.zeros((num_test_groups, 2), dtype=np.float64)
    mid_patch_top1_pred_utms = np.zeros((num_test_groups, 2), dtype=np.float64)
    mid_patch_pred_utms_with_weight = np.zeros((num_test_groups, 2), dtype=np.float64)

    all_preds_utm_centers = [center for group in groups for center in group.class_centers]
    all_preds_utm_centers = torch.tensor(all_preds_utm_centers).to(args.device)

    # queries_descriptors = torch.zeros(5, model.feature_dim).type(torch.float64)
    queries_descriptors = np.zeros((5, model.feature_dim), dtype=np.float64)





    ########## 前5召回率 distances_top5 ###############
    distances_top3_total = np.zeros((num_test_groups, 3), dtype=np.float64)
    distances_top5_total = np.zeros((num_test_groups, 5), dtype=np.float64)




    images_info_update = []
    
    with torch.no_grad():

        cache_filename = f"cache/{args.dataset_name}_M{args.M}_N{args.N}_mipc{args.min_images_per_class}_retrieval.torch" # ORIGION
        images_paths, utmeast_utmnorth, images_per_class, classes_ids = torch.load(cache_filename)
        # descriptors_gallery_path = f'dataloader/{args.dataset_name}_M{args.M}_N{args.N}_mipc{args.min_images_per_class}/bb_{args.backbone}_agg_{args.aggregator}_descriptors.npy'
        descriptors_gallery_path = f'dataloader/{args.dataset_name}_M{args.M}_N{args.N}_mipc{args.min_images_per_class}/descriptors.npy'
        descriptors_gallery = np.load(descriptors_gallery_path)
        # faiss_index = faiss.IndexFlatL2(model.feature_dim)
        # faiss_index.add(descriptors_gallery)

        # query_class are the UTMs of the center of the class to which the query belongs
        for query_i, (images_list, images_info) in enumerate(tqdm(test_dl, ncols=100)):
            images_list = images_list.squeeze(0)    # 否则images_list的大小为torch.Size([1, 5, 3, 336, 448])
            images_list = images_list.to(args.device)
            images_info = process_dict(images_info)
            query_utm = torch.tensor((images_info['utm_e'], images_info['utm_n'])).to(args.device) # 这个是中心点的utm
            pred_group_utms = []
            for i in range(len(images_list)):
                image = images_list[i].unsqueeze(0).to(args.device)
                descriptor = model(image)
                all_preds_confidences = torch.zeros([0], device=args.device)
                for j in range(len(classifiers)):
                    pred = compute_pred(classifiers[j], descriptor)
                    assert pred.shape[0] == 1  # pred has shape==[1, num_classes]
                    all_preds_confidences = torch.cat([all_preds_confidences, pred[0]])
                topn_pred_class_id = all_preds_confidences.argsort(descending=True)[:max(LR_N)]
                pred_class_id = all_preds_utm_centers[topn_pred_class_id]
                pred_group_utms.append(pred_class_id)
                pred_top1_groups[query_i, i] = pred_group_utms[i][0]    # 记录5个位置的图像预测的top1结果，每次记录5个utm，其实是class的中心坐标
                # images_info[i] = pred_class_id

                images_info[i] = tuple(pred_group_utms[i][0].tolist())

                queries_descriptors[i] = descriptor.cpu().numpy()


                all_preds_class_ids = [center for group in groups for center in group.classes_ids]
                all_preds_class_ids = torch.tensor(all_preds_class_ids).to(args.device)
                pred_class_id_real = all_preds_class_ids[topn_pred_class_id]
                
                pred_top3_class_ids[query_i, i] = pred_class_id_real[:3]
                pred_top1_class_ids[query_i, i] = pred_class_id_real[0]
                pred_top5_class_ids[query_i, i] = pred_class_id_real[:5]

                del pred_class_id_real


            group_retrieval_utms = np.zeros((5, 10, 2), dtype=np.float64)
            group_retrieval_utms_sample_weight = np.zeros((5, 10), dtype=np.float64)
            for images_idx in range(len(images_list)):
                top5_classes_images_list = []
                query_descriptor = queries_descriptors[i]
                query_descriptor = query_descriptor.astype(np.float32)
                for topk_idx in range(3):   # 这里确定使用前多少个class作为检索区域
                    class_id = pred_top5_class_ids[query_i, images_idx, topk_idx]
                    class_id = tuple(int(x) for x in class_id.tolist())
                    images_paths_current_class = images_per_class[class_id]
                    top5_classes_images_list.extend(images_paths_current_class)
                images_index_list = [images_paths.index(x) for x in top5_classes_images_list]
                top5_classes_utms_list = utmeast_utmnorth[images_index_list]
                database_descriptors = descriptors_gallery[images_index_list]
                faiss_index = faiss.IndexFlatL2(model.feature_dim)
                database_descriptors = database_descriptors.astype(np.float32)  # 添加这行
                faiss_index.add(database_descriptors)

                descriptors_distances, pred_indexes = faiss_index.search(query_descriptor.reshape(1, -1), 10)   # NOTE 两个矩阵array的大小都是(1,10)

                descriptors_distances[descriptors_distances == 0] = 1e-5  # 或者你认为合适的非零值，防止除以非零值
                # pred_sample_weight = 1 / (descriptors_distances + 1)
                pred_sample_weight = 1 / descriptors_distances

                retrieval_utms = top5_classes_utms_list[pred_indexes]

                group_retrieval_utms[images_idx] = retrieval_utms       # shape(1, 10, 2)

                group_retrieval_utms_sample_weight[images_idx] = pred_sample_weight

            cross_point = ransac_mid_point(group_retrieval_utms, None)
            cross_point_with_weight = ransac_mid_point(group_retrieval_utms, group_retrieval_utms_sample_weight)
            mid_patch_with_weight = get_cluster_center(group_retrieval_utms[0], group_retrieval_utms_sample_weight[0])
            
            ransac_pred_utms[query_i] = cross_point
            ransac_pred_utms_with_weight[query_i] = cross_point_with_weight


            mid_patch_top1 = group_retrieval_utms[0, 0]
            mid_patch_top1_pred_utms[query_i] = mid_patch_top1
            mid_patch_pred_utms_with_weight[query_i] = mid_patch_with_weight

            # from scipy.spatial.distance import cdist
            query_utm_arr = query_utm.cpu().numpy()
            dist_ransac = dist(cross_point, query_utm_arr)
            dist_ransac_with_weight = dist(cross_point_with_weight, query_utm_arr)
            dist_mid_patch = dist(mid_patch_top1, query_utm_arr)
            dist_mid_patch_with_weight = dist(mid_patch_with_weight, query_utm_arr)


            ransac_distances[query_i] = dist_ransac
            ransac_with_weight_distances[query_i] = dist_ransac_with_weight
            mid_patch_top1_distances[query_i] = dist_mid_patch
            mid_patch_with_weight_distances[query_i] = dist_mid_patch_with_weight
            
            
            dist_class=torch.cdist(query_utm.unsqueeze(0).to(torch.float64), pred_group_utms[0].to(torch.float64)).squeeze()  # NOTE 这里没有经过筛选，直接用了中心位置的utm作为估计图像的UTM
            valid_distances[query_i] = dist_class

            images_info_update.append(images_info)



            ############# 计算前5召回率 ################
            mid_patch_top5_utms = group_retrieval_utms[0, :5]
            mid_patch_top5_utms = mid_patch_top5_utms.reshape(5, 2)
            distances_top5 = np.linalg.norm(mid_patch_top5_utms - query_utm_arr, axis=1)
            distances_top5_total[query_i] = distances_top5
            

    classifiers = [c.cpu() for c in classifiers]
    torch.cuda.empty_cache()  # Release classifiers memory
    lr_ns = []
    for N in LR_N:
        lr_ns.append(torch.count_nonzero((valid_distances[:, :N] <= threshold).any(axis=1)).item() * 100 / num_test_groups)

    gcd_str = ", ".join([f'LR@{N}: {acc:.2f}' for N, acc in zip(LR_N, lr_ns)])

    def get_results(distances: np.ndarray, threshold):
        lr = np.sum(distances <= threshold) / distances.size * 100
        mean_down_threshold_value = mean_down_threshold(distances, threshold)
        return lr, mean_down_threshold_value

    ransac_distances_mean = ransac_distances.mean()
    ransac_with_weight_distances_mean = ransac_with_weight_distances.mean()
    mid_patch_top1_distances_mean = mid_patch_top1_distances.mean()
    mid_patch_with_weight_distances_mean = mid_patch_with_weight_distances.mean()

    retrieval_threshold = 100
    # retrieval_threshold = args.retrieval_threshold
    ransac_lr = np.sum(ransac_distances < retrieval_threshold) / num_test_groups * 100
    ransac_with_weight_lr = np.sum(ransac_with_weight_distances < retrieval_threshold) / num_test_groups * 100
    mid_patch_top1_lr = np.sum(mid_patch_top1_distances < retrieval_threshold) / num_test_groups * 100
    mid_patch_with_weight_lr = np.sum(mid_patch_with_weight_distances < retrieval_threshold) / num_test_groups * 100

    ransac_mean_down_threshold = mean_down_threshold(ransac_distances, retrieval_threshold)
    ransac_with_weight_mean_down_threshold = mean_down_threshold(ransac_with_weight_distances, retrieval_threshold)
    mid_patch_top1_mean_down_threshold = mean_down_threshold(mid_patch_top1_distances, retrieval_threshold)
    mid_patch_with_weight_mean_down_threshold = mean_down_threshold(mid_patch_with_weight_distances, retrieval_threshold)



    ############## 前5召回率 #####################

    mask = np.any(distances_top5_total < retrieval_threshold, axis=1)
    # 计算符合要求的组数
    num_groups_meeting_criteria = np.sum(mask)
    recall_top5 = num_groups_meeting_criteria / num_test_groups * 100
    ##############################################





    ransac_str = f'ransac distance mean: {ransac_distances_mean:.2f}, when threshold = {retrieval_threshold}, LR@1:{ransac_lr:.2f}, mean down threshold: {ransac_mean_down_threshold:.2f}'
    ransac_with_weight_str = f'ransac with weight distance mean: {ransac_with_weight_distances_mean:.2f}, when threshold = {retrieval_threshold}, LR@1:{ransac_with_weight_lr:.2f}, mean down threshold: {ransac_with_weight_mean_down_threshold:.2f}'
    mid_patch_top1_str = f'recall top 5: {recall_top5:.2f}, mid-patch top1 retrieval distance mean: {mid_patch_top1_distances_mean:.2f}, when threshold = {retrieval_threshold}, LR@1:{mid_patch_top1_lr:.2f}, mean down threshold: {mid_patch_top1_mean_down_threshold:.2f}'
    mid_patch_with_weight_str = f'mid-patch mean with weight retrieval distance mean:{mid_patch_with_weight_distances_mean:.2f}, when threshold = {retrieval_threshold}, LR@1:{mid_patch_with_weight_lr:.2f}, mean down threshold: {mid_patch_with_weight_mean_down_threshold:.2f}'

    retrieval_threshold = 50
    ransac_lr, ransac_mean_down_threshold = get_results(ransac_distances, retrieval_threshold)
    ransac_with_weight_lr, ransac_with_weight_mean_down_threshold = get_results(ransac_with_weight_distances, retrieval_threshold)
    mid_patch_top1_lr, mid_patch_top1_mean_down_threshold = get_results(mid_patch_top1_distances, retrieval_threshold)
    mid_patch_with_weight_lr, mid_patch_with_weight_mean_down_threshold = get_results(mid_patch_with_weight_distances, retrieval_threshold)

    ransac_str = ransac_str + f', when threshold = {retrieval_threshold}, LR@1:{ransac_lr:.2f}, mean down threshold: {ransac_mean_down_threshold:.2f}'
    ransac_with_weight_str = ransac_with_weight_str + f', when threshold = {retrieval_threshold}, LR@1:{ransac_with_weight_lr:.2f}, mean down threshold: {ransac_with_weight_mean_down_threshold:.2f}'
    mid_patch_top1_str = mid_patch_top1_str + f', when threshold = {retrieval_threshold}, LR@1:{mid_patch_top1_lr:.2f}, mean down threshold: {mid_patch_top1_mean_down_threshold:.2f}'
    mid_patch_with_weight_str = mid_patch_with_weight_str + f', when threshold = {retrieval_threshold}, LR@1:{mid_patch_with_weight_lr:.2f}, mean down threshold: {mid_patch_with_weight_mean_down_threshold:.2f}'

    return gcd_str, images_info_update, ransac_str, ransac_with_weight_str, mid_patch_top1_str, mid_patch_with_weight_str


import time
import psutil

# global perf flags (GPU)
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
def inference_time_count(args, model, classifiers, test_dl, groups, images_info, class_subset_count):
    gpu_memory_usage = []  # 记录显存使用
    cpu_memory_usage = []  # 记录CPU内存使用
    num_test = len(images_info) # 推理图像的数量
    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]

    # 添加pred_top5_class_ids的定义
    pred_top5_class_ids = torch.zeros(num_test, 5, 2).type(torch.int64)


    all_preds_utm_centers = [center for group in groups for center in group.class_centers]
    all_preds_utm_centers = torch.tensor(all_preds_utm_centers).to(args.device)
    # print(f"all_preds_utm_centers: {all_preds_utm_centers}")

    # 加载检索所需数据
    cache_filename = f"cache/{args.dataset_name}_M{args.M}_N{args.N}_mipc{args.min_images_per_class}_retrieval.torch"
    images_paths, utmeast_utmnorth, images_per_class, _ = torch.load(cache_filename)
    descriptors_gallery_path = f'dataloader/{args.dataset_name}_M{args.M}_N{args.N}_mipc{args.min_images_per_class}/descriptors.npy'
    descriptors_gallery = np.load(descriptors_gallery_path)

    # 统计时间
    classification_times = []
    retrieval_times = []
    
    distances_top5_total = np.zeros((num_test, 5), dtype=np.float64)
    distances_top1_total = np.zeros(num_test, dtype=np.float64)

    # warmup
    warmup_iters = min(5, len(images_info))
    for i, (image, _) in enumerate(test_dl):
        if i >= warmup_iters: break
        image = image.to(args.device)
        with torch.no_grad():  # 新增：禁用梯度计算
            _ = model(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # images_info_update = []
    with torch.no_grad():
        for query_i, (image, images_info) in enumerate(tqdm(test_dl, ncols=100)):
            # 获取推理前的显存
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     gpu_mem_before = torch.cuda.memory_allocated(args.device) / 1024 / 1024  # MB
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(args.device)

            # 获取加载前的内存
            process = psutil.Process()
            cpu_mem_before = process.memory_info().rss / 1024 / 1024  # MB
            # 记录切图时间
            # print(images_info)
            image = image.to(args.device)
            query_utm = torch.tensor((images_info['utm_e'], images_info['utm_n'])).to(args.device) # 这个是中心点的utm

            # 记录推理开始时间
            # inference_start = time.time()
            
            # 记录分类时间
            # class_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            descriptor = model(image)
            all_preds_confidences = torch.zeros([0], device=args.device)    # 创建一个长度为0的一维张量，相当于一个空张量
            for j in range(len(classifiers)):
                pred = compute_pred(classifiers[j], descriptor)
                all_preds_confidences = torch.cat([all_preds_confidences, pred[0]])
            
            topn_pred_class_id = all_preds_confidences.argsort(descending=True)[:max(LR_N)] # 取前五个class

            # 添加query_utm的定义
            query_utm = torch.tensor((images_info['utm_e'], images_info['utm_n'])).to(args.device)            
            
            # 添加检索步骤
            query_descriptor = descriptor.cpu().numpy().astype(np.float32)

            # 在分类部分添加pred_top5_class_ids的更新
            all_preds_class_ids = [center for group in groups for center in group.classes_ids]
            all_preds_class_ids = torch.tensor(all_preds_class_ids).to(args.device)
            pred_class_id_real = all_preds_class_ids[topn_pred_class_id]
            pred_top5_class_ids[query_i] = pred_class_id_real[:5]
            # classification_times.append(time.time() - class_start)
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                classification_times.append(start_event.elapsed_time(end_event) / 1000.0)  # seconds
            else:
                # CPU fallback
                class_start = time.time()
                _ = descriptor
                classification_times.append(time.time() - class_start)

            # 获取前class_subset_count个类别
            top5_classes_images_list = []
            ######################
    
            ######################
            retrieval_start = time.time()
            for topk_idx in range(class_subset_count):
                class_id = pred_top5_class_ids[query_i, topk_idx]
                class_id = tuple(int(x) for x in class_id.tolist())
                images_paths_current_class = images_per_class[class_id]
                top5_classes_images_list.extend(images_paths_current_class) # 整合前top5_classes_images_list对应subset的图像路径

            # 构建FAISS索引并检索
            images_index_list = [images_paths.index(x) for x in top5_classes_images_list]
            top5_classes_utms_list = utmeast_utmnorth[images_index_list]
            database_descriptors = descriptors_gallery[images_index_list]
            faiss_index = faiss.IndexFlatL2(model.feature_dim)
            database_descriptors = database_descriptors.astype(np.float32)
            faiss_index.add(database_descriptors)

            # 检索并计算距离
            _, pred_indexes = faiss_index.search(query_descriptor.reshape(1, -1), 4)
            retrieval_utms = top5_classes_utms_list[pred_indexes]
            retrieval_times.append(time.time()-retrieval_start)
            query_utm_arr = query_utm.cpu().numpy()
            
            # 计算前5个检索结果的距离
            mid_patch_top5_utms = retrieval_utms[0, :5]
            mid_patch_top1_utms = retrieval_utms[0, 0]
            distances_top1 = dist(mid_patch_top1_utms, query_utm_arr)
            distances_top5 = np.linalg.norm(mid_patch_top5_utms - query_utm_arr, axis=1)
            distances_top5_total[query_i] = distances_top5
            distances_top1_total[query_i] = distances_top1

            # if torch.cuda.is_available():
            #     gpu_mem_after = torch.cuda.memory_allocated(args.device) / 1024 / 1024  # MB
            #     gpu_memory_usage.append(gpu_mem_after - gpu_mem_before)

            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
                gpu_memory_usage.append(peak_mem)
            # 获取加载后的内存
            cpu_mem_after = process.memory_info().rss / 1024 / 1024  # MB
            cpu_memory_usage.append(cpu_mem_after - cpu_mem_before)
    # 计算结果
    retrieval_threshold = 100
    recall_top1 = np.sum(distances_top1_total < retrieval_threshold) / num_test * 100
    mask = np.any(distances_top5_total < retrieval_threshold, axis=1)
    recall_top5 = np.sum(mask) / num_test * 100
    avg_classification_time = np.mean(classification_times)
    avg_retrieval_time = np.mean(retrieval_times)
    avg_inference_time = avg_classification_time + avg_retrieval_time

    result_str = f'R@1: {recall_top1:.2f}, R@5: {recall_top5:.2f}\n'
    # result_str += f'Average cut time: {avg_cut_time:.4f}s\n'
    result_str += f'Average inference time: {avg_inference_time:.4f}s\n'
    result_str += f'Average classification time: {avg_classification_time:.4f}s\n'
    result_str += f'Average retrieval time: {avg_retrieval_time:.4f}s'

    return result_str, gpu_memory_usage, cpu_memory_usage
