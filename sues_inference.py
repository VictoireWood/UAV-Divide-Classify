import torch
from tqdm import tqdm
# import faiss
import numpy as np
import logging

from classifiers import LinearLayer

# EDIT
import parser
args = parser.parse_arguments()

LR_N = [1, 5, 10, 20]



def evaluate(qf, ql, gf, gl):   # query feature(当前的query feature), query label, gallery feature, gallery label
    # print(qf.shape) torch.Size([512])
    # print(gf.shape) torch.Size([51355, 512])
    # print(ql) 0 ()
    # print(gl) [0,0...0] len = 51355 shape = (51355,)

    query = qf.view(-1, 1)
    # print(query.shape)  query.shape = (512,1)
    # gf.shape = (51355, 512)
    # 矩阵相乘

    # score 是否可理解为当前余弦距离的排序？
    score = torch.mm(gf, query)
    # score.shape = (51355,1)
    score = score.squeeze(1).cpu()
    # score.shape = （51355,)
    score = score.numpy()
    # print(score)

    # predict index
    index = np.argsort(score)  # from small to large
    # 从小到大的索引排列
    # print("index before", index)
    index = index[::-1]
    # print("index after", index)
    # 从大到小的索引排列

    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    # print(query_index.shape) (54, 1)
    # gl = ql 返回标签值相同的索引矩阵
    # 得到 ql：卫星图标签，gl：无人机图标签
    # 即 卫星图标签在 gl中的索引位置 组成的矩阵
    good_index = query_index

    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)
    # print(junk_index)  = []

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    # CMC就是recall的，只要前K里面有一个正确答案就算recall成功是1否则是0
    # mAP是传统retrieval的指标，算的是 recall和precision曲线，这个曲线和x轴的面积。
    # 你可以自己搜索一下mAP

    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    # print(cmc.shape) torch.Size([51355])
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    # print(index.shape) (51355,)
    # if junk_index == []
    # return index fully

    # find good_index index
    ngood = len(good_index)
    # print("good_index", good_index) (54, 1)
    # print(index)
    # print(good_index)
    mask = np.in1d(index, good_index)
    # print(mask)
    # print(mask.shape)  (51355,)
    # 51355 中 54 个对应元素变为了True

    rows_good = np.argwhere(mask == True)
    # print(rows_good.shape) (54, 1)
    # rows_good 得到这 54 个为 True 元素的索引位置

    rows_good = rows_good.flatten()
    # print(rows_good.shape)  (54,)
    # print(rows_good[0])

    cmc[rows_good[0]:] = 1
    # print(cmc)
    # print(cmc.shape) torch.Size([51355])

    # print(cmc)
    for i in range(ngood):
        d_recall = 1.0 / ngood
        # d_racall = 1/54
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        # n/sum
        # print("row_good[]", i, rows_good[i])
        # print(precision)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def inference(args, model, drone_dl, satellite_dl, num_drone_images, num_satellite_images):
    
    model = model.eval()
    descriptors_dim = model.feature_dim  # 特征的维度
    queries_descriptors = torch.zeros((num_drone_images, descriptors_dim), dtype=torch.float32)
    queries_class_ids = torch.zeros(num_drone_images, dtype=torch.int16)
    database_descriptors = torch.zeros((num_satellite_images, descriptors_dim), dtype=torch.float32)
    database_class_ids = torch.zeros(num_satellite_images, dtype=torch.int16)
    
    with torch.no_grad():
        # query_class are the UTMs of the center of the class to which the query belongs
        for query_i, (images, class_id) in enumerate(tqdm(drone_dl, ncols=100)):
            images = images.to(args.device)
            descriptors = model(images)
            # descriptors = descriptors.cpu().numpy()
            descriptors = descriptors.cpu()
            # class_id = class_id.numpy()
            queries_descriptors[query_i, :] = descriptors
            queries_class_ids[query_i] = class_id


            
        for query_i, (images, class_id) in enumerate(tqdm(satellite_dl, ncols=100)):
            images = images.to(args.device)
            descriptors = model(images)
            # descriptors = descriptors.cpu().numpy()
            descriptors = descriptors.cpu()
            # class_id = class_id.numpy()
            database_descriptors[query_i, :] = descriptors
            database_class_ids[query_i] = class_id
        
    # faiss_index = faiss.IndexFlatL2(descriptors_dim)
    # faiss_index.add(database_descriptors)
    # del database_descriptors
    # logging.debug("Calculating recalls")
    # k = max(LR_N)

    # _, indexes =  faiss_index.search(queries_descriptors, k)    # indexes的shape应该是(queries的数量,k)

    # class_pred = database_class_ids[indexes]    # shape应该也是(queries的数量, k)
    # class_gt = queries_class_ids    # shape是(queries的数量)

    # class_gt = np.expand_dims(class_gt, 1)
                
    # torch.cuda.empty_cache()  # Release classifiers memory
    # lr_ns = []
    # for N in LR_N:
    #     class_pred_N = class_pred[:, :N]
    #     lr_count = np.sum(np.any(class_pred_N == class_gt, axis=1))
    #     lr_ns.append(lr_count.item() * 100 / num_drone_images)

    # gcd_str = ", ".join([f'LR@{N}: {acc:.1f}' for N, acc in zip(LR_N, lr_ns)])

    query_feature = queries_descriptors
    gallery_feature = database_descriptors
    query_label = queries_class_ids
    gallery_label = database_class_ids

        # fed tensor to GPU
    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    # CMC = recall
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    # ap = average precision
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp

    # average CMC

    CMC = CMC.float()
    CMC = CMC / len(query_label)

    # show result and save
    result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
        CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
        ap / len(query_label) * 100)
    
    return result


