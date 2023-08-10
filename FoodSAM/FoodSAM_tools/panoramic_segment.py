import csv
import random
import cv2
import numpy as np
import os
import json
from FoodSAM_tools.enhance_semantic_masks import predict_sam_label
def get_IoU(bbox_red, bbox_green):
    ix_min = max(bbox_red[0], bbox_green[0]) 
    iy_min = max(bbox_red[1], bbox_green[1]) 
    ix_max = min(bbox_red[2], bbox_green[2]) 
    iy_max = min(bbox_red[3], bbox_green[3]) 
   
    iw = max(ix_max - ix_min, 0.0)
    ih = max(iy_max - iy_min, 0.0)
    
    inters = iw * ih
    
    red_area = (bbox_red[2] - bbox_red[0]) * (bbox_red[3] - bbox_red[1]) 
    green_area = (bbox_green[2] - bbox_green[0]) * (bbox_green[3] - bbox_green[1]) 

    uni = red_area + green_area - inters 

    iou = inters / uni
    return iou


def background2category(data_folder, thresholds = [0.5, 0.6],
                     method_dir='object_detection',
                     background_dir = 'sam_mask_label',
                     metadata_path = 'sam_metadata.csv',
                     save_dir = 'background2category',
                     ):
    picture_ids = os.listdir(data_folder)
    for threshold in thresholds:
        for id in picture_ids:
            if id == 'sam_process.log':
                continue
            metadata_file = os.path.join(data_folder, id, metadata_path)
            meta = np.genfromtxt(metadata_file, delimiter=',', dtype=str)
            meta_columns = meta[0]
            meta_data = []
            for i in range(1, meta.shape[0]):
                item = dict()
                for j in range(meta.shape[1]):
                    item[meta_columns[j]] = meta[i][j]
                meta_data.append(item)

            methods = os.listdir(os.path.join(data_folder, id, method_dir))
            for method in methods:
                method_path = os.path.join(data_folder, id, method_dir, method)
                with open(method_path, 'r', encoding="utf-8") as f:
                    boxs = json.load(f)
                backgrounds = os.listdir(os.path.join(data_folder, id, background_dir))
                for background in backgrounds:
                    background_path = os.path.join(data_folder, id, background_dir, background)
                    parts = []
                    with open(background_path, 'r') as f:
                        line = f.readline()
                        columns = line.split(',')
                        columns[-1] = columns[-1].replace('\n','') #去掉结尾的回车

                        line = f.readline()
                        while line:
                            line = line.split(',')
                            line[-1] = line[-1].replace('\n','')
                            item = dict()
                            for i in range(len(columns)):
                                item[columns[i]] = line[i]
                            parts.append(item)
                            line = f.readline()
                
                    for i in range(len(parts)):
                        if parts[i]['category_name'] == 'background':
                            assert parts[i]['id'] == meta_data[i]['id']
                            best_iou = 0.0
                            index = -1
                            box = [float(meta_data[i]['bbox_x0']), float(meta_data[i]['bbox_y0']), 
                                   float(meta_data[i]['bbox_x0']) + float(meta_data[i]['bbox_w']), 
                                   float(meta_data[i]['bbox_y0']) + float(meta_data[i]['bbox_h'])]
                            for j in range(len(boxs)):
                                temp = get_IoU(box, boxs[j]['bounding_box'])
                                if temp >= best_iou:
                                    index = j
                                    best_iou = temp
                            if best_iou >= threshold:
                                parts[i]['category_name'] = boxs[index]['category_name']
                                parts[i]['category_id'] = str(int(boxs[index]['category_id']) + 104)
                    temp_dir = os.path.join(data_folder, id, save_dir,  method.split('.')[0], background.split('.')[0])
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    save_path = os.path.join(temp_dir,'threshold-' + str(threshold) + '.txt')
                    handle = open(save_path , mode='w')

                    for i in range(len(columns)):
                        handle.write(columns[i])
                        if i != len(columns) - 1:
                            handle.write(',')
                        else:
                            handle.write('\n')

                    for part in parts:
                        values = list(part.values())
                        for i in range(len(values)):
                            handle.write(str(values[i]))
                            if i != len(values) - 1:
                                handle.write(',')
                            else:
                                handle.write('\n')
                    handle.close()


def random_color():
    r = random.randint(120, 255)
    g = random.randint(120, 255) 
    b = random.randint(120, 255)
    return [b, g, r]

def visualization_save(mask, save_path, boxes, info, color_list, color_list2, num_class):
    font = cv2.FONT_HERSHEY_SIMPLEX
    mask = mask.astype(np.uint8)
    vis = mask.copy()
    for  id, coord in enumerate(boxes):
        x0, y0, w, h = coord[:4]
        label = info[id][0]
        pt1 = (int(x0), int(y0))
        pt2 = (int(x0+w), int(y0+h))
        color = color_list[int(label)].tolist()
        if label > num_class-1: 
            color = color_list2[int(label)].tolist()
        cv2.rectangle(vis, pt1, pt2, color, 2)
        x, y = pt1
        category_name = info[id][1]
        cv2.putText(vis, category_name, (x+3, y+10), font, 0.35, color, 1)
    cv2.imwrite(save_path, vis)  


def panoramic_segment(data_folder, category_txt, color_list_path, num_class=104,  area_thr=0.01, ratio_thr=0.5, top_k=80,
                  masks_path_name="sam_mask/masks.npy",
                  panoramic_mask_name='panoramic_vis.png', 
                  instance_mask_name='instance_vis.png',
                  method_dir='object_detection', 
                  metadata_path = 'sam_metadata.csv',
                  new_label_save_dir = 'background2category',
                  method='od_UniDet',
                  sam_mask_label_file_name='sam_mask_label.txt',
                  pred_mask_file_name='pred_mask.png',
                  sam_mask_label_file_dir='sam_mask_label'
                  ):
    
    
    predict_sam_label([data_folder], category_txt, masks_path_name, sam_mask_label_file_name, pred_mask_file_name, sam_mask_label_file_dir)
    
    
    background2category(data_folder,
                        method_dir=method_dir,
                        background_dir = sam_mask_label_file_dir,
                        metadata_path = metadata_path,
                        save_dir = new_label_save_dir)
    
    color_list = np.load(color_list_path)
    color_list2 = color_list[::-1]
    color_list[0] = [238, 239, 20]
    
    for img_folder in os.listdir(data_folder):
        if img_folder == 'sam_process.log':
            continue
        category_info_path = os.path.join(data_folder, img_folder, new_label_save_dir, method, sam_mask_label_file_dir, 'threshold-0.5.txt')
        instance_category_info_path = os.path.join(data_folder, img_folder, sam_mask_label_file_dir, sam_mask_label_file_name)
        
        sam_mask_folder = os.path.join(data_folder, img_folder)
        csv_path = os.path.join(data_folder, img_folder, metadata_path)
        img_path = os.path.join(data_folder, img_folder, 'input.jpg')
        h,w = cv2.imread(img_path).shape[0], cv2.imread(img_path).shape[1]
        save_dir = os.path.join(data_folder, img_folder)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, panoramic_mask_name)

        f = open(category_info_path, 'r')
        category_info = f.readlines()[1:]
        f.close()

        f = open(instance_category_info_path, 'r')
        instance_category_info = f.readlines()[1:]
        f.close()

        category_info = sorted(category_info, key=lambda x:float(x.split(',')[4]), reverse=True)
        category_info = category_info[:top_k]
        enhanced_mask = np.zeros((h, w, 3))
        
        box_cats = {}
        sam_masks = np.load(os.path.join(sam_mask_folder, masks_path_name))
        for info in category_info:
            idx, label, count_ratio, area = info.split(',')[0], int(info.split(',')[1]), float(info.split(',')[3]), float(info.split(',')[4])
            if area < area_thr and label < num_class:
                continue
            if count_ratio < ratio_thr:
                continue
            sam_mask = sam_masks[int(idx)].astype(bool)
            assert (sam_mask.sum()/ (sam_mask.shape[0] * sam_mask.shape[1]) - area) < 1e-4  

            if label == 0:
                continue
            elif label < num_class:
                enhanced_mask[sam_mask] = random_color()
            else:
                enhanced_mask[sam_mask] = color_list[label]
            if label != 0:
                box_cats[idx] = (label, info.split(',')[2], area)

        boxes = []
        info = []
        with open(csv_path) as f:
            rows = csv.reader(f)
            for row in rows:
                if row[0] in box_cats:
                    label = box_cats[row[0]][0]
                    name = box_cats[row[0]][1]
                    area = box_cats[row[0]][2]
                    boxes.append([float(row[2]), float(row[3]), float(row[4]), float(row[5]),area])
                    info.append((label, name))
        boxes= np.array(boxes)
        visualization_save(enhanced_mask, save_path, boxes, info, color_list, color_list2, num_class)
        
        #instance segmentation
        box_cats = {}
        instance_category_info = sorted(instance_category_info, key=lambda x:float(x.split(',')[4]), reverse=True)
        instance_category_info = instance_category_info[:top_k]
        enhanced_mask = np.zeros((h, w, 3))
        save_path = os.path.join(save_dir, instance_mask_name)
        for info in instance_category_info:
            idx, label, count_ratio, area = info.split(',')[0], int(info.split(',')[1]), float(info.split(',')[3]), float(info.split(',')[4])
            if area < area_thr and label < num_class:
                continue
            if count_ratio < ratio_thr:
                continue
            sam_mask = sam_masks[int(idx)].astype(bool)
            assert (sam_mask.sum()/ (sam_mask.shape[0] * sam_mask.shape[1]) - area) < 1e-4  

            if label == 0:
                continue
            elif label < num_class:
                enhanced_mask[sam_mask] = random_color()  
            else:
                enhanced_mask[sam_mask] = color_list[label]
            if label != 0:
                box_cats[idx] = (label, info.split(',')[2], area)

        boxes = []
        info = []
        with open(csv_path) as f:
            rows = csv.reader(f)
            for row in rows:
                if row[0] in box_cats:
                    label = box_cats[row[0]][0]
                    name = box_cats[row[0]][1]
                    area = box_cats[row[0]][2]
                    boxes.append([float(row[2]), float(row[3]), float(row[4]), float(row[5]),area])
                    info.append((label, name))
        boxes= np.array(boxes)
        visualization_save(enhanced_mask, save_path, boxes, info, color_list, color_list2, num_class)

