import cv2
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
import os
def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label] 
    area_intersect, _ = np.histogram( 
        intersect, bins=np.arange(num_classes + 1)) 
    area_pred_label, _ = np.histogram(  
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect 

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=float)
    total_area_union = np.zeros((num_classes, ), dtype=float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=float)
    total_area_label = np.zeros((num_classes, ), dtype=float)
    for i in range(num_imgs):
      
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)

        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label


    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label

def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
  
    all_acc = total_area_intersect.sum() / total_area_label.sum() 
    acc = total_area_intersect / total_area_label                
    ret_metrics = [all_acc, acc]                            
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union 
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics

def evaluate(output_folder, ann_folder, num_class, metric='mIoU', logger=None, efficient_test=False, ignore_index=255,
             pred_mask_name = "enhance_mask.png"):


    file_ids = os.listdir(output_folder)
    file_ids = [x for x in file_ids if os.path.exists(os.path.join(ann_folder,x+'.png') )]
    file_ids = sorted(file_ids, key=lambda x:int(x))
    gt_mask = []
    results = []
    for x in file_ids:
        gt_mask_path = os.path.join(ann_folder, x +'.png')
        mask = cv2.imread(gt_mask_path)
        mask = mask[:, :, 2]
        pred_mask_path = os.path.join(output_folder, x, pred_mask_name)
        pred_mask = cv2.imread(pred_mask_path, 0)
        assert pred_mask.shape == mask.shape
        gt_mask.append(mask)
        results.append(pred_mask)

    class_names = [str(i) for i in range(num_class)]

    assert len(gt_mask) == len(results)
    for idx, name in enumerate(gt_mask):
        assert gt_mask[idx].shape == results[idx].shape        

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        
        ret_metrics = eval_metrics(
            results,
            gt_mask,
            num_class,
            ignore_index,
            metric,
            label_map=None, 
            reduce_zero_label=False) 

        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']] 
        ret_metrics_round = [  
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_class): 
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']] 
        ret_metrics_mean = [  
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):  
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0 
        if mmcv.is_list_of(results, str): 
            for file_name in results:
                os.remove(file_name)
        return eval_results
