#import tensorflow as tf
import general_processing
#import trained_faster_ir2
import context as ct
import os

def main(_):
    # -*- coding: UTF-8 -*-

    if ct.on_cloud:
        """
        print('0')
        os.system('unzip -d /cos_person/275/1745/simpledet/ /cos_person/275/1745/simplenet/simpledet-master.zip')
        os.system('unzip -d /cos_person/275/1745/simpledet/ /cos_person/275/1745/simplenet/cocoapi-master.zip')
        os.system('unzip -d /cos_person/275/1745/simpledet/ /cos_person/275/1745/simplenet/incubator-mxnet-master.zip')
        os.system('cp -r /cos_person/275/1745/simpledet/simpledet-master/operator_cxx/* /cos_person/275/1745/simpledet/incubator-mxnet-master/src/operator')
        os.system('mkdir -p /cos_person/275/1745/simpledet/incubator-mxnet-master/src/coco_api')
        os.system('cp -r /cos_person/275/1745/simpledet/cocoapi-master/common /cos_person/275/1745/simpledet/incubator-mxnet-master/src/coco_api')
        print('1')
        os.system('''cd /cos_person/275/1745/simpledet/incubator-mxnet-master;
                  echo "USE_SIGNAL_HANDLER = 1" >> ./config.mk;
                  echo "USE_OPENCV = 0" >> ./config.mk;
                  echo "USE_MKLDNN = 0" >> ./config.mk;
                  echo "USE_BLAS = openblas" >> ./config.mk;
                  echo "USE_CUDA = 1" >> ./config.mk;
                  echo "USE_CUDA_PATH = /usr/local/cuda" >> ./config.mk;
                  echo "USE_CUDNN = 1" >> ./config.mk;
                  echo "USE_NCCL = 1" >> ./config.mk;
                  echo "USE_DIST_KVSTORE = 1" >> ./config.mk;
                  echo "CUDA_ARCH = -gencode arch=compute_61,code=sm_61" >> ./config.mk;
                  make -j$((`nproc`-1));''')

        print('2')
        os.system('python3 /cos_person/275/1745/simpledet/incubator-mxnet-master/python/setup.py install')
        """
        #os.system('unzip -d /cos_person/275/1745/mmdetection_raw/ /cos_person/275/1745/mmdetection/mmdetection.zip')
        #os.system("""cd /cos_person/275/1745/mmdetection/mmdetection_raw;
        #             compile.sh;
        #             pip install -e .""")
        #os.system('singularity shell --no-home --nv -s /usr/bin/zsh --bind / /cos_person/275/1745/object_detection/simpledet.img')
        #os.system('unzip -d / /cos_person/275/1745/object_detection/train_03')
        #general_processing.rezip('train_05', 30000)
        #general_processing.unzip()
        #general_processing.get_MD5('/cos_person/275/1745/object_detection/train_08.zip')
        #for i in range(1, 11):
        #   general_processing.get_MD5('/cos_person/275/1745/object_detection/train_08.z%02d'%(i))
        
        os.system('pip install pandas')
        input_prefix = '/cos_person/275/1745/object_detection/output/'
        prefix = '/cos_person/275/1745/object_detection/backup/submit/'
        general_processing.correct_submit_csv(input_prefix+'sample_submission.csv', 
                                              prefix+'950x950_bz2_bn_lr0.002_e0-60_corrected.csv')
        general_processing.complete_submit_file(prefix+'950x950_bz2_bn_lr0.002_e0-60_corrected.csv', 
                                                prefix+'950x950_bz2_bn_lr0.002_e0-60_corrected_completed.csv')
        """
        general_processing.zipDir('/cos_person/275/1745/mmdetection/mmdetection', 
                                  '/cos_person/275/1745/object_detection/mmdet_cus.zip')
        """
    else:
        '''
        #os.system('python "D:/Backup/Documents/Visual Studio 2015/Projects/object_detection/openimages2coco_master/convert.py" -p "D:/Backup/Documents/Visual Studio 2015/object_detection"')
        import pandas as pd
        data_root = 'D:/Backup/Documents/Visual Studio 2015/object_detection/validation-annotations-bbox.csv'
        a = pd.read_csv(data_root)
        b = a[a['ImageID']=='000595fe6fee6369']
        for i in range(len(b)):
            c = b[i:i+1][['XMin', 'YMin', 'XMax', 'YMax']].values
            d = b[i:i+1]['LabelName'].values
        #a = trained_faster_ir2.output_list_by_pbfile('D:\\Backup\\Documents\\Visual Studio 2015\\Projects\\object_detection\\test')
        #a = general_processing.sample_judger('D:/Backup/Documents/Visual Studio 2015/object_detection/train-annotations-bbox.csv', 
        #                                     'D:/Backup/Downloads/class-descriptions-boxable.csv')
        #a.jugde('dec30d5b87f69003')
        '''
        """
        import pickle
        with open("output/test2.pkl", "rb") as f:
            output = pickle.load(f)
        
        import torch
        pred = torch.Tensor([[1, 1, 3, 3], [2, 2, 3, 3]])
        target = torch.Tensor([[2, 2, 4, 4], [3, 3, 4, 4]])
        diff = torch.abs(pred - target)
        beta = 1
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                        diff - 0.5 * beta)
        
        import pickle
        # for debugging
        with open('output/cls_score', 'rb') as f:
            cls_score = pickle.load(f)
        with open('output/labels', 'rb') as f:
            labels = pickle.load(f)
        with open('output/label_weights', 'rb') as f:
            label_weights = pickle.load(f)
        with open('output/bbox_weights', 'rb') as f:
            bbox_weights = pickle.load(f)
        a = bbox_weights.tolist()
        b = label_weights.tolist()
        
        """
        sample_prefix = 'D:/Backup/Documents/Visual Studio 2015/object_detection/'
        file_prefix = 'C:/Users/Administrator/Desktop/kaggle_output/'
        csv_list = [file_prefix+'submit_0-10000.csv', file_prefix+'submit_10000-20000.csv',
                    file_prefix+'submit_20000-25000.csv', file_prefix+'submit_25000-30000.csv',
                    file_prefix+'submit_30000-40000.csv', file_prefix+'submit_40000-50000.csv',
                    file_prefix+'submit_50000-60000.csv', file_prefix+'submit_60000-70000.csv',
                    file_prefix+'submit_70000-75000.csv', file_prefix+'submit_75000-80000.csv',
                    file_prefix+'submit_80000-90000.csv', file_prefix+'submit_90000-99999.csv']
        general_processing.merge_submit_file_pieces(sample_prefix+'sample_submission.csv',
                                                    csv_list, file_prefix+'total_output.csv')
        """
        prefix = 'D:/Backup/Documents/Visual Studio 2015/Projects/object_detection/output/TF/'
        general_processing.submit_thresholding(prefix+'TF_output_completed_corrected.csv',
                                               prefix+'TF_output_completed_corrected_th07.csv',
                                               0.65)
        """
    k = 1
   
if __name__ == '__main__':
    main(0)