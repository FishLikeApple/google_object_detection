import torch
import numpy as np
import pandas as pd
import json
from ...core.bbox.geometry import bbox_overlaps

num2class_file = '/cos_person/275/1745/object_detection/class-descriptions-boxable.csv'
num2class_csv = pd.read_csv(num2class_file)
label2id = {cat: i+1 for i, cat in enumerate(num2class_csv['Id'])}

comloss = 'type3'

# hierarchy preprocessing
def get_class_belonging(current_note, class_belonging, label2id):
    '''get the class belonging list in a recursive manner, note that class_belonging[0] is of subcategory, and
       class_belonging[1] is of parts.'''

    if current_note["LabelName"] == '/m/0cgh4':  # building is special
        current_note["Part"] = [{"LabelName": "/m/02dgv", "Part": [{"LabelName": "/m/03c7gz"}]},
                                {"LabelName": "/m/04m6gz"}, 
                                {"LabelName": "/m/01lynh"},
                                {"LabelName": "/m/0d4v4"},
                                {"LabelName": "/m/031b6r"}]

    subcategory_list = []
    if "Subcategory" in current_note:
        for child in current_note["Subcategory"]:
            get_class_belonging(child, class_belonging, label2id)
            subcategory_list.append(label2id[child["LabelName"]])
            subcategory_list.extend(class_belonging[0][label2id[child["LabelName"]]])

    part_list = []
    if "Part" in current_note:
        for part in current_note["Part"]:
            get_class_belonging(part, class_belonging, label2id)
            part_list.append(label2id[part["LabelName"]])
            part_list.extend(class_belonging[1][label2id[part["LabelName"]]])
            part_list.extend(class_belonging[0][label2id[part["LabelName"]]])  # subcategories of parts of one are also parts of the one

    if current_note["LabelName"] in label2id:
        class_belonging[0][label2id[current_note["LabelName"]]] = subcategory_list
        class_belonging[1][label2id[current_note["LabelName"]]] = part_list

    return class_belonging

hierarchy_file_path_name = '/cos_person/275/1745/object_detection/bbox_labels_600_hierarchy.json'
hierarchy = json.load(open(hierarchy_file_path_name, encoding='utf-8'))   
# set to 99999 for debugging
class_belonging = [list(np.ones([602])*99999), list(np.ones([602])*99999)]
class_belonging[0][0] = []
class_belonging[1][0] = []
class_belonging = get_class_belonging(hierarchy, class_belonging, label2id)
print('class_belonging:')
for one in class_belonging:
    print(one)
'''
# cooccurrence preprocessing
class_cooccurrence = []
for label in self.num2class_csv['Id']:
    if label in ['Human eye', 'Skull', 'Head', 'Human face', 'Human mouth', 'Human ear', 'Human head', 'Hair', 
                 'Human hand', 'Human foot', 'Human arm', 'Human leg', 'Human beard', 'Cowboy hat', 'Dress',
                 'Fedora', 'Footwear', 'Sandal', 'Sports uniform', 'Coat', 'Glasses', 'Helmet', 'Jeans',
                 'High heels', 'Scarf', 'Earrings', 'Bicycle helmet']:
        class_cooccurrence.append([label2id['Man'], label2id['Woman'], label2id['Boy'], 
                                   label2id['Girl'], label2id['Person']])
    elif label in ['Human eye', 'Skull', 'Head', 'Human face', 'Human mouth', 'Human ear', 'Human head', 'Hair', 
                 'Human hand', 'Human foot', 'Human arm', 'Human leg', 'Human beard']:
        class_cooccurrence.append([label2id['Man'], label2id['Woman'], label2id['Boy'], 
                                   label2id['Girl'], label2id['Person']])
    else:
        class_cooccurrence.append([])
'''
def calculate_loss_weights(cls_score, bbox_pred, labels, bbox_targets, flags, 
                           label_weights, bbox_weights, loss_type=comloss, debug=False):
    """calculate new loss weights, note that label_weights and 
       bbox_weights are original weights (which may be all 1)."""
    
    min_containing_rate = 0.9

    if loss_type == 'type2':
        false_positive_balance_rate = 0.8
        containing_balance_rate = 0.1
    elif loss_type == 'type1':
        false_positive_balance_rate = 0.5
        containing_balance_rate = 1
    elif loss_type == 'type3':
        false_positive_balance_rate = 0.8
        containing_balance_rate = 1

    cls_pred_list = torch.argmax(cls_score, dim=1).tolist()
    bbox_pred_list = bbox_pred.tolist()
    labels_list = labels.tolist()
    bbox_targets_list = bbox_targets.tolist()

    # group absolution
    if flags != None:
        label_absolution_list = labels[flags==1].tolist()
    else:
        label_absolution_list = []

    for i in range(len(labels_list)):
        # group absolution
        if labels_list[i] in label_absolution_list:
            label_weights[i] *= 0
            bbox_weights[i] *= 0
        elif cls_pred_list[i] != labels_list[i]:
            # belonging
            if labels_list[i] in class_belonging[0][cls_pred_list[i]]:
                label_weights[i] *= 0
                bbox_weights[i] *= 0
            
            # false positive balance
            elif labels_list[i] == 0:
                label_weights[i] *= false_positive_balance_rate
                bbox_weights[i] *= false_positive_balance_rate
            # containing
            elif containing_balance_rate < 1:
                pos_inds = labels > 0
                for j in range(len(labels_list)):
                    if debug:
                        print('cls_pred_list[i]')
                        print(cls_pred_list[i])
                        print('class_belonging[1][labels_list[j]]')
                        print(class_belonging[1][labels_list[j]])
                        print('')

                    if (labels_list[j]>0) and (cls_pred_list[i] in class_belonging[1][labels_list[j]]):
                        overlap_rates = bbox_overlaps(bbox_pred[i].unsqueeze(0), bbox_targets[pos_inds], mode='iof')[0].tolist()

                        if debug:
                            print('overlap_rates:')
                            print(bbox_pred[i].unsqueeze(0))
                            print(bbox_targets[pos_inds])
                            print(overlap_rates)

                        for k in range(len(overlap_rates)):
                            if overlap_rates[k] >= min_containing_rate:
                                label_weights[i] *= containing_balance_rate
                                bbox_weights[i] *= containing_balance_rate

                        """
                        if (bbox_pred[i][2]<=bbox_targets[j][0]) or (bbox_targets[i][2]<=bbox_pred[j][0]):
                            pass
                        elif (bbox_pred[i][3]<=bbox_targets[j][1]) or (bbox_targets[i][3]<=bbox_pred[j][1]):
                            pass
                        else:
                            # X
                            if bbox_pred[i][0] <= bbox_targets[j][0]:
                                if bbox_pred[i][2] <= bbox_targets[j][2]:
                                    wid = bbox_pred[i][2] - bbox_targets[j][0]
                                else:
                                    wid = bbox_targets[j][2] - bbox_targets[j][0]
                            else:
                                if bbox_targets[j][2] <= bbox_pred[i][2]:
                                    wid = bbox_targets[j][2] - bbox_pred[i][0]
                                else:
                                    wid = bbox_pred[i][2] - bbox_pred[i][0]
                            #Y
                            if bbox_pred[i][1] <= bbox_targets[j][1]:
                                if bbox_pred[i][3] <= bbox_targets[j][3]:
                                    wid = bbox_pred[i][3] - bbox_targets[j][1]
                                else:
                                    wid = bbox_targets[j][3] - bbox_targets[j][1]
                            else:
                                if bbox_targets[j][3] <= bbox_pred[i][1]:
                                    wid = bbox_targets[j][3] - bbox_pred[i][1]
                                else:
                                    wid = bbox_pred[i][3] - bbox_pred[i][1]
                        """
            
            
    #label_weights = 
    #bbox_weights[i] *= weight

    return label_weights, bbox_weights

def loss_test():
    "test if the loss function works right"
 
    for loss_type in ['type1', 'type2', 'type3']:
        cls_score = np.ones([1024, 602]) * (1.0/1024)
        for i in range(1024):
            cls_score[i][0] = 0.5
        cls_score[10][2] = 0.8
        cls_score[11][53] = 0.8
        cls_score[12][53] = 0.8
        cls_score[13][1] = 0.8
        cls_score = torch.Tensor(cls_score)

        bbox_pred = np.ones([1024, 4]) * 0.3
        for i in range(1024):
            bbox_pred
        bbox_pred[10] = [0.25, 0.0, 0.75, 0.5]
        bbox_pred[11] = [0.5, 0.5, 1.0, 1.0]
        bbox_pred[12] = [0.5, 0.25, 1.0, 0.75]
        bbox_pred[13] = [0, 0.5, 0.5, 1.0]
        bbox_pred = torch.Tensor(bbox_pred)

        labels = np.zeros([1024])
        labels[10] = 170
        labels[11] = 40
        labels[12] = 40
        labels = torch.LongTensor(labels)
             
        bbox_targets = np.ones([1024, 4]) * 0.0
        bbox_targets[10] = [0.0, 0.0, 0.5, 0.5]
        bbox_targets[11] = [0.5, 0.5, 1.0, 1.0]
        bbox_targets[12] = [0.5, 0.0, 1.0, 0.5]
        bbox_targets = torch.Tensor(bbox_targets)

        flags = None

        label_weights = np.ones([1024]) 
        label_weights = torch.Tensor(label_weights)

        bbox_weights = np.ones([1024])
        bbox_weights = torch.Tensor(bbox_weights)
    
        label_weight_output, bbox_weight_output = calculate_loss_weights(cls_score, bbox_pred, labels, bbox_targets, flags, 
                                                                         label_weights, bbox_weights, loss_type, debug=True)
        print(loss_type+':')
        for i, weight in enumerate(label_weight_output.tolist()):
            if weight != 1:
                print('[i, weight]:')
                print([i, weight])
        print('')

