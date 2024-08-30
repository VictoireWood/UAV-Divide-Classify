
import torch
from tqdm import tqdm

from classifiers import LinearLayer

# EDIT
import parser
args = parser.parse_arguments()

from generate_database import M_list, flight_heights


LR_N = [1, 5, 10, 20]
# threshold = 25  # ORIGION 原来的M=20m，这里设置的是偏移25米算成功召回
# threshold = 500 # EDIT 这里是要配合M的设定，M设为800米
threshold = args.threshold  # EDIT

def compute_pred(criterion, descriptors):
    if isinstance(criterion, LinearLayer):
        # Using LinearLayer
        return criterion(descriptors, None)[0]
    else:
        # Using AMCC/LMCC
        return torch.mm(descriptors, criterion.weight.t())


#### Validation
def inference(args, model, classifiers, test_dl, groups, num_test_images):
    
    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]
    valid_distances = torch.zeros(num_test_images, max(LR_N))

    # EDIT 加上可变threshold的初始化
    threshold_list_total = torch.zeros(num_test_images, max(LR_N))
        
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
            pred_class_id = all_preds_utm_centers[topn_pred_class_id]    # NOTE 这里能不能把从classid里得到自适应M然后得到threshold
            
            # ORIGION
            # dist=torch.cdist(query_utms.unsqueeze(0), pred_class_id.to(torch.float64))
            # valid_distances[query_i] = dist

            # EDIT
            if threshold is None:
                Heights = list(map(int, pred_class_id[:,0].tolist())) # 将list中的数值全部变为int类型，作为索引
                M_indexes = [flight_heights.index(h) for h in Heights]
                M_params = [M_list[idx] for idx in M_indexes]
                threshold_list = M_params
            else:
                threshold_list = [threshold] * max(LR_N)    # 所有元素都相同

            dist = torch.cdist(query_utms.unsqueeze(0), pred_class_id.to(torch.float64))
            valid_distances[query_i] = dist
            threshold_list_total[query_i] = torch.tensor(threshold_list)

    classifiers = [c.cpu() for c in classifiers]
    torch.cuda.empty_cache()  # Release classifiers memory
    lr_ns = []
    for N in LR_N:
        # lr_ns.append(torch.count_nonzero((valid_distances[:, :N] <= 25).any(axis=1)).item() * 100 / num_test_images)    # ORIGION  计算20个cell有多少成功匹配的
        # lr_ns.append(torch.count_nonzero((valid_distances[:, :N] <= threshold).any(axis=1)).item() * 100 / num_test_images) # EDIT 把判定阈值给改了

        # EDIT
        valid_list = []
        for img_idx in range(num_test_images):
            valid_topn = [valid_distances[img_idx, n] <= threshold_list_total[img_idx, n] for n in range(N)]
            valid_list.append(valid_topn) # EDIT 改成可以自适应的阈值
        lr_ns.append(torch.count_nonzero(torch.tensor(valid_list).any(axis=1)).item() * 100 / num_test_images)

    gcd_str = ", ".join([f'LR@{N}: {acc:.1f}' for N, acc in zip(LR_N, lr_ns)])
    
    return gcd_str

