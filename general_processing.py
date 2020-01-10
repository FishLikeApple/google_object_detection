import os
try:
    import tensorflow as tf
    on_tensorflow = True
except:
    on_tensorflow = False
import numpy as np

import glob
def get_file_paths_from_folder(folder, extensions=["jpg", "jpeg"]):
    """like the name"""

    file_paths = []

    for extension in extensions:
        file_glob = glob.glob(folder+"/*."+extension)  #不分大小写
        file_paths.extend(file_glob)   #添加文件路径到file_paths

    return file_paths 

#def add_water_point(image, point):

#def overlay_image():

def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.01)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.01)

    return tf.clip_by_value(image, 0., 1.)

def distort_grayscale_image(image):
    distorted_image = tf.image.grayscale_to_rgb(image)
    distorted_image = distort_color(distorted_image)
    distorted_image = tf.image.rgb_to_grayscale(distorted_image)
    distorted_image = tf.image.grayscale_to_rgb(distorted_image)
    return distorted_image

def distort_rgb_image(image):
    distorted_image = distort_color(image)
    #random = tf.random_uniform([1], 0, 2, tf.int32)
    #distorted_image = tf.cond(tf.equal(random, 0),
    #                  lambda:tf.image.rgb_to_grayscale(distorted_image),
    #                  lambda:distorted_image)
    return distorted_image

def preprocessing_for_training(image, unified_image_shape, bbox=None, augmentation=True):
    distorted_image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 查看是否存在标注框。
    if bbox is not None:
        """
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(distorted_image), bounding_boxes=bbox, min_object_covered=0.9)
        """
        bbox_begin0 = tf.expand_dims(bbox[1], 0)
        bbox_begin1 = tf.expand_dims(bbox[0], 0)
        bbox_begin2 = tf.expand_dims(bbox[1]*0, 0)
        bbox_begin = tf.concat([bbox_begin0, bbox_begin1, bbox_begin2], 0)

        bbox_size0 = tf.expand_dims(bbox[3]-bbox[1], 0)
        bbox_size1 = tf.expand_dims(bbox[2]-bbox[0], 0)
        bbox_size2 = tf.expand_dims(tf.shape(distorted_image)[2], 0)
        bbox_size = tf.concat([bbox_size0, bbox_size1, bbox_size2], 0)

        distorted_image = tf.slice(distorted_image, bbox_begin, bbox_size)

    #data augmentation
    if augmentation == True:
        distorted_image = tf.random_crop(
            distorted_image, [tf.cast(tf.cast(tf.shape(distorted_image)[0], tf.float32)*0.95, tf.int32), 
                              tf.cast(tf.cast(tf.shape(distorted_image)[1], tf.float32)*0.95, tf.int32),
                              tf.shape(distorted_image)[2]])   #①

        distorted_image = tf.cond(tf.equal(tf.shape(distorted_image)[2], 1), 
                                 lambda: distort_grayscale_image(distorted_image),
                                 lambda: distort_rgb_image(distorted_image))   
        distorted_image = tf.image.random_flip_left_right(distorted_image)
    else:
        distorted_image = tf.cond(tf.equal(tf.shape(distorted_image)[2], 1), 
                                 lambda: tf.image.grayscale_to_rgb(distorted_image),
                                 lambda: distorted_image)   

    if unified_image_shape != None:
        # standardization, which is divided into 2 parts, expansion and padding
        image_shape = tf.cast(tf.shape(distorted_image), tf.float64)
        shape_ratio1 = unified_image_shape[0] / image_shape[0] 
        shape_ratio2 = unified_image_shape[1] / image_shape[1]
        shape_one = tf.cast(image_shape[0:2]*shape_ratio2, tf.int32)
        shape_two = tf.cast(image_shape[0:2]*shape_ratio1, tf.int32)
        resized_image = tf.cond(shape_ratio1>shape_ratio2, 
                                lambda:tf.image.resize_images(distorted_image, shape_one),
                                lambda:tf.image.resize_images(distorted_image, shape_two))

        resized_image = tf.image.resize_image_with_crop_or_pad(resized_image, unified_image_shape[0], 
                                                               unified_image_shape[1])
    else:
        resized_image = distorted_image

    return resized_image

import queue
if on_tensorflow:
    from tensorflow.python.platform import gfile
class batch_generator:
    ''' a class just for remembering parameters'''
    def __init__(self, file_paths, file_labels, unified_image_shape, bboxes=None, batch_size=32):
        self.file_paths = file_paths 
        self.file_labels = file_labels 
        self.unified_image_shape = unified_image_shape
        self.bboxes = bboxes
        self.batch_size = batch_size
        self.data_queue = queue.Queue(maxsize=20)

        #define an entrance net
        graph = tf.Graph()
        self.sess = tf.Session(graph=graph)
        with graph.as_default():
            self.raw_image_input = tf.placeholder(tf.string)
            #self.bbox_input = tf.placeholder(tf.int32, [4])  # later changing
            if self.bboxes == None:
                self.bbox_input = None
            image = tf.image.decode_jpeg(self.raw_image_input)
            self.image_output = preprocessing_for_training(image, self.unified_image_shape, self.bbox_input)
            self.image_output_without_augmentation = preprocessing_for_training(
                image, self.unified_image_shape, self.bbox_input, False)

    def get_specific_batch(self, image_path_list, augmentation=False):
        """like it's name."""

        data_batch = []

        if augmentation == False:
            output_tensor = self.image_output_without_augmentation
        else:
            output_tensor = self.image_output

        for path in image_path_list:
            if self.bboxes == None:
                data_batch.append(self.sess.run(output_tensor,
                                                {self.raw_image_input:gfile.FastGFile(path, "rb").read()}))
            else:
                data_batch.append(self.sess.run(output_tensor,
                                                {self.raw_image_input:gfile.FastGFile(path, "rb").read(),
                                                    self.bbox_input:self.bboxes[image_path_list.index(path)]}))

        return data_batch

# submission helper functions
def get_prediction_string(result):
    """from each result, generates the complete prediction string in the format {Label Confidence XMin YMin XMax YMax},{...} based on submission file."""
    prediction_strings = []
    for index, score in enumerate(result['detection_scores']):
        index = int(index)
        single_prediction_string = ""
        single_prediction_string += result['detection_class_names'][index].decode("utf-8") + " "  + str(score) + " "
        single_prediction_string += " ".join(str(x) for x in result['detection_boxes'][index])
        prediction_strings.append(single_prediction_string)

    prediction_string = ", ".join(str(x) for x in prediction_strings)
    return prediction_string

def get_prediction_entry(filepath, result):
    return {
        "ImageID": get_image_id_from_path(filepath),
        "PredictionString": get_prediction_string(result)
    }

import gc
def unzip():
    """unzip files on cloud"""
    """
    import threading
    main_path  = '/cos_person/275/1745/object_detection/'
    unzip_threads = []
    unzip_file_names = ['train_00']
    for unzip_file_name in unzip_file_names:
        unzip_threads.append(threading.Thread(target=os.system, 
                                                args=('unzip -d /cos_person/275/1745/object_detection/train_images/ /'+main_path+unzip_file_name,)))
        unzip_threads[-1].daemon = True
        unzip_threads[-1].start()
    # wait for unzip
    for unzip_thread in unzip_threads:
        unzip_thread.join()
    """
    print('start to unzip')
    file_names = ['train_07', 'train_08']
    for file_name in file_names:
        gc.collect()
        os.system('cat /cos_person/275/1745/object_detection/'+file_name+'.z* > '+file_name+'_all.zip')
        #os.system('zip -F train_00_all.zip --out fixed.zip')
        os.system('unzip -d / /'+file_name+'_all.zip')
        print('unzip is over, start to rezip')
        #os.system('cp /'+file_name+' /cos_person/275/1745/object_detection/'+file_name)
        rezip(file_name, 30000, '/')

    #print('start to upload')
    #os.system('cp /train_00 /cos_person/275/1745/object_detection/train_00')
    #os.system('unzip -d /validation/ /cos_person/275/1745/object_detection/validation')
    #os.system('unzip -d /train_02/ /cos_person/275/1745/object_detection/train_02')
    #os.system('unzip -d /submit_test/ /cos_person/275/1745/object_detection/submit_test.zip')
    '''
    print('start to unzip')
    os.system('unzip -d /cos_person/275/1745/object_detection/ /cos_person/275/1745/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.zip')
    '''
    
import zipfile
def zip_list(path_name_list, output_path_name):
    """
    压缩指定路径+名list中文件
    :param path_name_list: 目标文件路径+名
    :param output_path_name: 压缩文件保存路径+名
    :return: 无
    """

    zip = zipfile.ZipFile(output_path_name, "w", zipfile.ZIP_DEFLATED)
    for path_name in path_name_list:
        zip.write(path_name)
    zip.close()

def zipDir(dirpath,outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """

    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath,'')

        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

import threading
def rezip(rezipped_file_name, new_num_images, 
          main_path='/cos_person/275/1745/object_detection/'):
    """rezip files on cloud"""

    print('start copy_and_unzip in rezip')
    copy_and_unzip(main_path+rezipped_file_name, '/train_images/')

    print('start mkdir and rezip_threads')
    file_paths = get_file_paths_from_folder('/train_images/'+rezipped_file_name+'/')
    rezip_threads = []
    os.system('mkdir /zip_output')
    for i in range(len(file_paths)//new_num_images):
        list_to_zip = []
        for j in range(new_num_images):
            list_to_zip.append(file_paths.pop(np.random.randint(0, len(file_paths))))

        zip_list(list_to_zip, '/zip_output/'+rezipped_file_name+'_'+str(i)+'.zip')
    # deal with remanents
    zip_list(file_paths, '/zip_output/'+rezipped_file_name+'_'+str(i+1)+'.zip')

    print('copy outputs to cloud')
    # copy outputs to cloud
    os.system('cp -r /zip_output/ /cos_person/275/1745/object_detection/sub_train/')

    # delete all caches
    os.system('rm  -rf /train_images/')
    os.system('rm  -rf /zip_output/')

def get_MD5(file_path_name='/cos_person/275/1745/object_detection/test'):
    """get a MD5 code of a file"""

    os.system('md5sum ' + file_path_name)

class sample_judger:
    """This is a class to judge whether to add a image."""

    def __init__(self, label_csv_path_name, map_csv_path_name, label_types=601):

        import pandas as pd
        self.label_csv = pd.read_csv(label_csv_path_name)
        self.counting_csv = pd.read_csv(map_csv_path_name)
        self.counting_csv.index = list(range(label_types))
        self.counting_csv.index += 1
      
        self.image_ID_list = []

    def jugde(self, labels):
        # jugde if you can add an image

#        for label in labels:
#            self.counting_csv.iloc[:,0] -(np.mean(self.counting_csv['counting']/np.mean(self.counting_csv['counting']))]
        return True

    def judge_and_add(self, image_ID, jugde=False):
        # try to add an image

        labels = self.label_csv.loc[self.label_csv['ImageID']==image_ID, 'LabelName']

        # return if the image isn't qualified
        if jugde and self.jugde(labels)!=True:
            return

        self.image_ID_list.append(image_ID)
        self.counting_csv[self.counting_csv.iloc[:,0].isin(labels)]['counting'] += 1

import threading
def package_installer(package_names, upgrade=True):
    """install needed packages"""

    if upgrade:
        postfix_text = '--upgrade '
    else:
        postfix_text = ''

    for package_name in package_names:
        
        # set for certain packages
        if package_name == 'mmcv':  # debug
            os.system("pip install "+postfix_text+package_name)
        else:
            os.popen("pip install "+postfix_text+package_name).read()

def install_packages(package_names, parallel=True, upgrade=True):
    """creat a thread to install needed packages"""

    threads = []
    if parallel:
        for package_name in package_names:
            threads.append(threading.Thread(target=package_installer, args=([package_name], upgrade)))
            threads[-1].daemon = True
            threads[-1].start()
        for thread in threads:
            thread.join()
    else:
        package_installer(package_names)

def quiet_system(command):
    """quiet os.system()"""
    #print('command:'+command+'starts')
    os.popen(command).read()
    #print('command:'+command+'is over')
    
def copy_and_unzip(original_path_name, target_path):
    # copy and unzip a zip file

    quiet_system('cp '+original_path_name+' /')
    quiet_system('unzip -d '+target_path+' /'+os.path.basename(original_path_name))

def load_and_unzip_train_data(file_prefix, unzip_file_names):
    """just for object detection now"""

    unzip_threads = []
    for unzip_file_name in unzip_file_names:
        quiet_system('cp '+file_prefix+unzip_file_name+' /')
        folder = os.path.splitext(unzip_file_name)[0]
        target = quiet_system('unzip -d /'+folder+'/ '+'/'+unzip_file_name)
    return unzip_threads

def get_loss_weights(sample_labels, gt_labels, sample_bboxes, gt_bboxes,
                    label_hierarchy, label_cooccurrence):
    # get loss weight of a sample
    
    weight = 1
    #for i in range(sample_labels.shape[0]):


    return weight
    
class data_preprocessor:
    """this class is used for preprocessing/filtering dataset and 
       generate a info csv file for current dataset."""

    def __init__(self,
                 img_prefix,
                 output_csv_path_name,
                 garbage_image_pk_path_name,
                 num2class_file='/cos_person/275/1745/object_detection/class-descriptions-boxable.csv',
                 ann_file='/cos_person/275/1745/object_detection/train-annotations-bbox.csv',
                 num_threads=8,
                 img_scale=(1024, 1024),
                 img_norm_cfg=dict(mean=[123.675, 116.28, 103.53], 
                                   std=[58.395, 57.12, 57.375], to_rgb=True),
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False):

        import pandas as pd
        from random import shuffle
        import pickle
        self.pd = pd
        self.shuffle = shuffle
        self.pickle = pickle

        self.num2class_csv = self.pd.read_csv(num2class_file)
        CLASSES = self.num2class_csv['Name']
        self.CLASSES_IDs = self.num2class_csv['Id']
        self.id2label = {cat: i+1 for i, cat in enumerate(self.CLASSES_IDs)}

        # images paths, note that it's a list due to multiple datasets
        self.img_prefix = img_prefix

        self.ann_file = ann_file

        # set sampling configs
        if output_csv_path_name == None:
            self.output_csv_path_name = img_prefix + '/image_ann.csv'
        else:
            self.output_csv_path_name = output_csv_path_name
        self.garbage_image_pk_path_name = garbage_image_pk_path_name
        self.num_threads = num_threads
        self.label_counting = np.zeros(len(self.id2label)+1)
        self.raw_label_counting = np.zeros(len(self.id2label)+1)
        self.highest_class_proportion = 1.0 / 75.0  # this proportion is (bbox number / image number)
        self.lowest_class_proportion = self.highest_class_proportion / 25.0  # this proportion is (bbox number / image number)
        self.enough_class = []
        self.lacking_class = list(range(1, len(self.label_counting)-1))
        self.label_index_mapping = []
        self.class_tolerance = 0
        self.class_craving = 1 # note that it's actually in an inverse ratio to craving
        self.garbage_path_name_list = []

    def load_annotations(self):
        return self.pd.read_csv(self.ann_file)

    def generate_csv(self):
        """generate a csv info file of the dataset as needed"""
        
        # run other code if objective files exist
        try:
            with open(self.output_csv_path_name, 'rb'):
                with open(self.garbage_image_pk_path_name, 'rb') as f:
                    garbage_path_name_list = self.pickle.load(f)
                    for garbage_path_name in garbage_path_name_list:
                        os.system('rm -rf '+garbage_path_name)
        except:
            # load annotations
            self.img_infos = self.load_annotations()

            self.image_list = []
            extensions = ["jpg", "jpeg"]
            for extension in extensions:
                file_glob = glob.glob(self.img_prefix+"/*."+extension)  #不分大小写
                self.image_list.extend(file_glob)   #添加文件路径到file_list

            # achieve data balance
            self.shuffle(self.image_list)
            self.highest_class_amount = len(self.image_list) * self.highest_class_proportion
            self.lowest_class_amount = len(self.image_list) * self.lowest_class_proportion
            sampling_threads = []
            self.output_DFs = []
            piece_len = len(self.image_list) // self.num_threads
            for i in range(self.num_threads):
                self.output_DFs.append(self.pd.DataFrame(columns=self.img_infos.columns))
                if i < self.num_threads-1:
                    iter_range = range(piece_len*i, piece_len*(i+1))
                else:
                    iter_range = range(piece_len*i, len(self.image_list))
                sampling_threads.append(threading.Thread(target=self.sequentially_sample_images, 
                                                      args=(i, iter_range,)))
                sampling_threads[-1].daemon = True
                sampling_threads[-1].start()

            # wait for unzip
            for i, sampling_thread in enumerate(sampling_threads):
                sampling_thread.join()

            with open("/cos_person/275/1745/object_detection/output/logs.txt", "a") as f:
                print('output_DFs lens:', file=f)
                for output_DF in self.output_DFs:
                    print(len(output_DF), file=f)
                print('', file=f)

            # save the output csv and garbage path names
            self.pd.concat(self.output_DFs).to_csv(self.output_csv_path_name)
            with open(self.garbage_image_pk_path_name, 'wb') as f:
                self.pickle.dump(self.garbage_path_name_list, f)
        
    def get_labels_and_flags(self, img_info, delete_origin=False):
        # like the name

        ImageID = os.path.splitext(img_info['filename'])[0] #test
        if delete_origin:
            ann_pieces_origin = self.img_infos[self.img_infos['ImageID']==ImageID]
            ann_pieces = ann_pieces_origin.copy()
            del ann_pieces_origin
        else:
            ann_pieces = self.img_infos[self.img_infos['ImageID']==ImageID]

        bboxes = []
        labels = []
        flags = []
        for i in range(len(ann_pieces)):
            labels.append(self.id2label[ann_pieces[i:i+1]['LabelName'].values[0]])
            flags.append(ann_pieces[i:i+1][['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']].values[0])

        labels = np.array(labels)
        flags = np.array(flags, ndmin=2)

        ann = dict(labels=labels.astype(np.int64),
                   flags=flags.astype(np.int64))
        return ann, ann_pieces

    def sequentially_sample_images(self, worker_num, iteration_range):
        """sample images to achieve balance"""

        # first iteration
        for i in iteration_range:
            img_info = {}
            img_info['filename'] = os.path.basename(self.image_list[i])
            ann, ann_pieces = self.get_labels_and_flags(img_info, delete_origin=True)
            # set mapping
            #for label in ann['labels']:
            #    self.label_index_mapping[label].append(i)
            # set label_counting
            if ((len(set(ann['labels']).intersection(set(self.enough_class)))<=self.class_tolerance)and(1 not in ann['flags'][:,2]))\
               or (len(set(ann['labels']).intersection(set(self.lacking_class)))>=self.class_craving): 
                """
                with open("/cos_person/275/1745/object_detection/output/logs.txt", "a") as f:
                    print('before append:', file=f)
                    print(len(self.output_DFs[worker_num]), file=f)
                    print(len(ann_pieces), file=f)
                    print(worker_num, file=f)
                """
                self.output_DFs[worker_num] = self.pd.concat([self.output_DFs[worker_num], ann_pieces])
                """
                with open("/cos_person/275/1745/object_detection/output/logs.txt", "a") as f:
                    print('after append:', file=f)
                    print(len(self.output_DFs[worker_num]), file=f)
                    print(worker_num, file=f)
                    print('', file=f)
                """
                for label in ann['labels']: 
                    self.label_counting[label] += 1
                    if self.label_counting[label] >= self.highest_class_amount:
                        if label not in self.enough_class:
                            self.enough_class.append(label)
                    if self.label_counting[label] >= self.lowest_class_amount:
                        if label in self.lacking_class:
                            self.lacking_class.remove(label)
            else:
                os.system('rm -rf '+self.image_list[i])
                self.garbage_path_name_list.append(self.image_list[i])

            # for debugging
            for label in ann['labels']:
                self.raw_label_counting[label] += 1
        
        """
        # second iteration, unused
        if self.lowest_class_amount > 0:
            while len(self.lacking_class) != 0:
                min_class = np.where(np.min(self.label_counting))
                index = sample(self.label_index_mapping[min_class], 1)
                img_info = {}
                img_info['filename'] = os.path.basename(self.image_list[index[0]])
                ann = self.get_labels_and_flags(img_info)
                for label in ann['labels']: 
                    self.label_counting[label] += 1
                    if self.label_counting[label] >= self.lowest_class_amount:
                        if label in self.lacking_class:
                            self.lacking_class.remove(label)
        """
        # for debugging
        with open("/cos_person/275/1745/object_detection/output/label_counting", "wb") as f:
            self.pickle.dump(self.label_counting, f)

        with open("/cos_person/275/1745/object_detection/output/raw_label_counting", "wb") as f:
            self.pickle.dump(self.raw_label_counting, f)
           
def object_result_to_string(shape, result, classes):
    """get string type output"""

    output = ''
    first = True
    for i, bboxes in enumerate(result):
        for bbox in bboxes:
            if first:
                first = False
            else:
                output = output + ' '
            if shape != None:
                output = output + str(classes[i]) + ' ' + str(bbox[4]) + ' ' + str(bbox[0]/shape[1])\
                   + ' ' + str(bbox[1]/shape[0]) + ' ' + str(bbox[2]/shape[1]) + ' ' + str(bbox[3]/shape[0])
            else:
                output = output + str(classes[i]) + ' ' + str(bbox[4]) + ' ' + str(bbox[0])\
                   + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3])
    return output

def parse_object_string(string):
    """get a formatted result list of a string item"""

    try:
        string_list = string.split(' ')
    except:
        return [], []

    ID_list = []
    bbox_list = []
    for i in range(len(string_list)):
        if (i%6) == 0:
            bbox_list.append([])
            ID_list.append(string_list[i])
        else:
            bbox_list[-1].append(float(string_list[i]))
                
    return ID_list, bbox_list

def restore_object_string(ID_list, bbox_list):
    """get a string item of a formatted result list"""

    string = ''
    for i in range(len(ID_list)):
        if i < (len(ID_list)-1):
            string = string + ID_list[i] + ' ' + str(bbox_list[i][0]) + ' ' + str(bbox_list[i][1])  + ' '+ \
                str(bbox_list[i][2])  + ' ' + str(bbox_list[i][3]) + ' ' + str(bbox_list[i][4]) + ' '
        else:
            string = string + ID_list[i] + ' ' + str(bbox_list[i][0]) + ' ' + str(bbox_list[i][1]) + ' ' + \
                str(bbox_list[i][2]) + ' ' + str(bbox_list[i][3]) + ' ' + str(bbox_list[i][4])
                
    return string

def merge_submit_files(file_path_name_list, output_path_name, soft_nms_func, class_weight_list=None):
    """merge multiple submission files into one"""

    import pandas as pd
    
    # open csv files
    file_list = []
    for file_path_name in file_path_name_list:
        file_list.append(pd.read_csv(file_path_name))

    # analyze
    ID_list = []
    bbox_list = []
    for i in range(len(file_list[0])):
        for file in file_list:
            prediction_string = file.loc[file['ImageId']==file_list[0].loc[i, 'ImageId'], 'PredictionString'].values[0]
            IDs, bboxes = parse_object_string(prediction_string)
            ID_list.extend(IDs)
            bbox_list.extend(bboxes)
        fixed_bboxes = soft_nms_func(np.array(bbox_list, dtype=np.float32), iou_thr=0.5, min_score=0.05)
        file_list[0].loc[i, 'ImageId'] = restore_object_string(ID_list, fixed_bboxes)

    # save csv output
    file_list[0].to_csv(output_path_name, index=False)

def complete_submit_file(file_path_name, output_path_name, 
                         num2class_file='/cos_person/275/1745/object_detection/class-descriptions-boxable.csv'):
    """complete a submission file with the missing label problem"""

    from mmdet.models.bbox_heads.customized_loss_weights import class_belonging
    import pandas as pd

    num2class_csv = pd.read_csv(num2class_file)
    # note that at some places label2id is id2num below
    id2num = {cat: i+1 for i, cat in enumerate(num2class_csv['Id'])}
    num2id = {i+1: cat for i, cat in enumerate(num2class_csv['Id'])}

    # open csv file
    file = pd.read_csv(file_path_name)

    for i in range(len(file)):
        IDs_to_add = []
        bboxes_to_add = []
        prediction_string = file.loc[i, 'PredictionString']
        IDs, bboxes = parse_object_string(prediction_string)
        for ID_i, ID in enumerate(IDs):
            for class_i in range(len(class_belonging[0])):
                if id2num[ID] in class_belonging[0][class_i]:
                    IDs_to_add.append(num2id[class_i])
                    bboxes_to_add.append(bboxes[ID_i])
        IDs.extend(IDs_to_add)
        bboxes.extend(bboxes_to_add)
        file.loc[i, 'PredictionString'] = restore_object_string(IDs, bboxes)

    # save csv output
    file.to_csv(output_path_name, index=False)

def correct_submit_csv(input_csv_path_name, output_csv_path_name, error_type='conf_last'):
    '''correct shift csv output'''

    import pandas as pd

    # open csv file
    input_csv = pd.read_csv(input_csv_path_name)

    # start to correct
    for i in range(len(input_csv)):
        prediction_string = input_csv.loc[i, 'PredictionString']
        IDs, bboxes = parse_object_string(prediction_string)
        for j in range(len(bboxes)):
            if error_type == 'conf_last':
                conf = bboxes[j][4]
                bboxes[j][1:5] = bboxes[j][0:4]
                bboxes[j][0] = conf
            elif error_type == 'disorder':
                bboxes[j] = [bboxes[j][3], bboxes[j][4], bboxes[j][0], bboxes[j][1], bboxes[j][2]]
        input_csv.loc[i, 'PredictionString'] = restore_object_string(IDs, bboxes)

    # save output
    input_csv.to_csv(output_csv_path_name, index=False)

def correct_TF_submit_csv(input_csv_path_name, output_csv_path_name):
    '''correct TF csv output'''

    import pandas as pd

    # open csv file
    input_csv = pd.read_csv(input_csv_path_name)

    # start to correct
    for i in range(len(input_csv)):
        prediction_string = input_csv.loc[i, 'PredictionString']
        IDs, bboxes = parse_object_string(prediction_string)
        for j in range(len(bboxes)):
            bboxes[j][1:5] = [bboxes[j][2], bboxes[j][1], bboxes[j][4], bboxes[j][3]]
        input_csv.loc[i, 'PredictionString'] = restore_object_string(IDs, bboxes)

    # save output
    input_csv.to_csv(output_csv_path_name, index=False)

def merge_submit_file_pieces(sample_path_name, input_path_name_list, output_path_name):
    """merge kaggle output"""

    import pandas as pd

    # open files
    sample_csv = pd.read_csv(sample_path_name)
    input_csv_list = []
    for input_path_name in input_path_name_list:
        input_csv_list.append(pd.read_csv(input_path_name))

    vacancy = 0
    # start to merge
    for i in range(len(sample_csv)):
        prediction = None
        for input_csv in input_csv_list:
            a = sample_csv.loc[i, 'ImageId']
            cache = input_csv.loc[input_csv['ImageId']==sample_csv.loc[i, 'ImageId'], 'PredictionString'].values
            for cc in cache:
                if cc!='0.0':
                    if prediction != None:
                        print('cache, i:')
                        print(cache)
                        print(i)
                        print('')
                    prediction = cc
        if prediction == None:
            print('cache, i(last):')
            print(cache)
            print(i)
            print('')
            vacancy += 1
        sample_csv.loc[i, 'PredictionString'] = prediction

    # save output
    sample_csv.to_csv(output_path_name, index=False)
    print('vacancy:')
    print(vacancy)

def submit_thresholding(input_path_name, output_path_name, threshold=0.7):
    '''threshold a submit file for a higher score'''

    import pandas as pd

    # open csv file
    input_csv = pd.read_csv(input_path_name)

    # start to correct
    for i in range(len(input_csv)):
        prediction_string = input_csv.loc[i, 'PredictionString']
        IDs, bboxes = parse_object_string(prediction_string)
        for j in range(len(bboxes)):
            if bboxes[j][0] >= threshold:
                bboxes[j][0] = 1.0
        input_csv.loc[i, 'PredictionString'] = restore_object_string(IDs, bboxes)

    # save output
    input_csv.to_csv(output_path_name, index=False)

def submit_scaling(input_path_name, output_path_name, interval=[0.06, 0.4]):
    '''scaling a submit file for a higher score'''

    import pandas as pd

    # open csv file
    input_csv = pd.read_csv(input_path_name)

    # start to correct
    for i in range(len(input_csv)):
        prediction_string = input_csv.loc[i, 'PredictionString']
        IDs, bboxes = parse_object_string(prediction_string)
        for j in range(len(bboxes)):
            if bboxes[j][0]>=interval[0] and bboxes[j][0]<=interval[1]:
                bboxes[j][0] *= 1/interval[1]
            elif bboxes[j][0] > interval[1]:
                bboxes[j][0] = 1.0
        input_csv.loc[i, 'PredictionString'] = restore_object_string(IDs, bboxes)

    # save output
    input_csv.to_csv(output_path_name, index=False)
