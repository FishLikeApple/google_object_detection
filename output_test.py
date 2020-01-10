from mmdet.apis import init_detector, inference_detector, show_result
import pandas as pd
import argparse 
import general_processing

def debugging_test():
    from mmdet.ops.nms.nms_wrapper import soft_nms
    import numpy as np
    a = soft_nms(np.array([[0.5, 0.5, 0.75, 0.75, 0.6], [0.5, 0.5, 1, 1, 0.4]], dtype=np.float32), iou_thr=0.5, min_score=0.05)
    print(a)

def test2():

    from mmdet.ops.nms.nms_wrapper import soft_nms

    input_prefix = '/cos_person/275/1745/object_detection/output/'
    main_path = '/cos_person/275/1745/object_detection/'
    prefix = '/cos_person/275/1745/object_detection/backup/submit/'
    file_path_name_list = [prefix+'950x950_bz2_lr0.002_comloss1_e0-19.csv',
                           prefix+'950x950_bz2_lr0.002_e0-19.csv']
    output_path_name = prefix+'950x950_bz2_lr0.002_comloss1_e0-19+950x950_bz2_lr0.002_e0-19.csv'
    #general_processing.merge_submit_files(file_path_name_list, output_path_name, soft_nms)
    csv_list = [prefix+'TF_output/TF_submission_1_0-15000.csv',
                prefix+'TF_output/TF_submission_1_15000-30000.csv',
                prefix+'TF_output/TF_submission_1_30000-45000.csv',
                prefix+'TF_output/TF_submission_1_45000-60000.csv',
                prefix+'TF_output/TF_submission_1_60000-70000.csv',
                prefix+'TF_output/TF_submission_1_70000-80000.csv',
                prefix+'TF_output/TF_submission_1_80000-90000.csv',
                prefix+'TF_output/TF_submission_1_90000-99999.csv',]

    #general_processing.merge_submit_file_pieces(main_path+'sample_submission.csv',
    #                                            csv_list, prefix+'TF_output.csv')
    #general_processing.correct_submit_csv(input_prefix+'950x950_bz2_bn_lr0.002_e0-60.csv', 
    #                                        prefix+'950x950_bz2_bn_lr0.002_e0-60_corrected.csv', error_type='disorder')
    general_processing.submit_scaling(input_prefix+'950x950_bz2_bn_lr0.002_comloss3_e0-60.csv', 
                                            prefix+'950x950_bz2_bn_lr0.002_comloss3_e0-60_scaled.csv')
    #general_processing.complete_submit_file(prefix+'950x950_bz2_bn_lr0.002_e0-60_corrected.csv', 
    #                                        prefix+'950x950_bz2_bn_lr0.002_e0-60_corrected_completed.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--iter_num', help='int')

    return parser.parse_args()

def main():
    
    from mmdet.models.bbox_heads.customized_loss_weights import loss_test
    loss_test()
    
    #class_belonging = get_class_belonging()
    """
    args = parse_args()

    num2class_file = '/cos_person/275/1745/object_detection/class-descriptions-boxable.csv'
    num2class_csv = pd.read_csv(num2class_file)
    classes = []
    for cat in num2class_csv['Id']:
        classes.append(cat) 

    mmdetection_path = '/cos_person/275/1745/mmdetection/mmdetection/'
    main_path  = '/cos_person/275/1745/object_detection/'
    config_file = main_path + 'code/dcn_best.py'
    output_path = "/cos_person/275/1745/object_detection/output/"
    #checkpoint_file = mmdetection_path + 'checkpoints/cascade_mask_rcnn_r16_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b86027a6.pth'
    checkpoint_file = output_path + "950x950_bz2_bn_lr0.002_e0-60.pth"
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a list of images and write the results to image files
    with open(output_path+'logs.txt', 'a') as f:
        print('testing', file=f)
    imgs = [mmdetection_path+'test1.jpg', mmdetection_path+'test2.jpg']
    for i, result in enumerate(inference_detector(model, imgs)):
        for j, bbox in enumerate(result):
            if len(bbox) > 0:
                with open(output_path+'logs.txt', 'a') as f:
                    print('label:', file=f)
                    print(j, file=f)
                    print('bbox:', file=f)
                    print(bbox, file=f)
                    print('', file=f)
        
        show_result(imgs[i], result, classes, score_thr=0, out_file=output_path+'result_'+str(i)+str(args.iter_num)+'.jpg')
        
                general_processing.merge_submit_file_pieces(sample_prefix+'sample_submission.csv',
                                                    csv_list, prefix+'TF_output.csv')
        """

if __name__ == '__main__':
    main()