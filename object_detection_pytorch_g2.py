#mode = 'training'
mode = 'generating'
#mode = 'debug_test'
#mode = 'test'

parts_per_dataset = 3
generating_start_num = 10
random_seed = 1745

def main(_):
    import os
    
    # install the basic pack
    os.system('pip install pandas')

    import general_processing
    import threading
    import sys
    import shutil
    import datetime

    mmdetection_path = '/cos_person/275/1745/mmdetection/mmdetection/'
    sys.path.append(mmdetection_path)
    main_path  = '/cos_person/275/1745/object_detection/'
    sampling_record_path = main_path + 'sub_train/data_sampling_records2/'

    quiet_training_iters = 2  # not used
    already_trained_iters = 0
    total_training_iters = 20

    # load and unzip data
    if (mode=='training') or (mode=='generating'):
        file_prefix = main_path + 'sub_train/zip_output/'
        unzip_file_names = ['train_00_0.zip', 'train_00_1.zip', 'train_00_2.zip', 'train_00_3.zip', 
                            'train_00_4.zip', 'train_00_5.zip', 'train_00_6.zip',
                            'train_01_0.zip', 'train_01_1.zip', 'train_01_2.zip', 'train_01_3.zip', 
                            'train_01_4.zip', 'train_01_5.zip', 'train_01_6.zip',
                            'train_02_0.zip', 'train_02_1.zip', 'train_02_2.zip', 'train_02_3.zip', 
                            'train_02_4.zip', 'train_02_5.zip', 'train_02_6.zip',
                            'train_03_0.zip', 'train_03_1.zip', 'train_03_2.zip', 'train_03_3.zip', 
                            'train_03_4.zip', 'train_03_5.zip', 'train_03_6.zip',
                            'train_04_0.zip', 'train_04_1.zip', 'train_04_2.zip', 'train_04_3.zip', 
                            'train_04_4.zip', 'train_04_5.zip', 'train_04_6.zip',
                            'train_05_0.zip', 'train_05_1.zip', 'train_05_2.zip', 'train_05_3.zip', 
                            'train_05_4.zip', 'train_05_5.zip', 'train_05_6.zip',
                            'train_06_0.zip', 'train_06_1.zip', 'train_06_2.zip', 'train_06_3.zip', 
                            'train_06_4.zip', 'train_06_5.zip', 'train_06_6.zip',
                            'train_07_0.zip', 'train_07_1.zip', 'train_07_2.zip', 'train_07_3.zip', 
                            'train_07_4.zip', 'train_07_5.zip', 'train_07_6.zip',
                            'train_08_0.zip', 'train_08_1.zip', 'train_08_2.zip', 'train_08_3.zip', 
                            'train_08_4.zip']
        
        # set training image paths corresponding to the above list
        unzip_prefix_list = ['/train_00_0/train_images/train_00/', '/train_00_1/train_images/train_00/',
                             '/train_00_2/train_images/train_00/', '/train_00_3/train_images/train_00/',
                             '/train_00_4/train_images/train_00/', '/train_00_5/train_images/train_00/',
                             '/train_00_6/train_images/train_00/',
                             '/train_01_0/train_images/train_01/', '/train_01_1/train_images/train_01/',
                             '/train_01_2/train_images/train_01/', '/train_01_3/train_images/train_01/',
                             '/train_01_4/train_images/train_01/', '/train_01_5/train_images/train_01/',
                             '/train_01_6/train_images/train_01/',
                             '/train_02_0/train_images/train_02/', '/train_02_1/train_images/train_02/',
                             '/train_02_2/train_images/train_02/', '/train_02_3/train_images/train_02/',
                             '/train_02_4/train_images/train_02/', '/train_02_5/train_images/train_02/',
                             '/train_02_6/train_images/train_02/',
                             '/train_03_0/train_images/train_03/', '/train_03_1/train_images/train_03/',
                             '/train_03_2/train_images/train_03/', '/train_03_3/train_images/train_03/',
                             '/train_03_4/train_images/train_03/', '/train_03_5/train_images/train_03/',
                             '/train_03_6/train_images/train_03/',
                             '/train_04_0/train_images/train_04/', '/train_04_1/train_images/train_04/',
                             '/train_04_2/train_images/train_04/', '/train_04_3/train_images/train_04/',
                             '/train_04_4/train_images/train_04/', '/train_04_5/train_images/train_04/',
                             '/train_04_6/train_images/train_04/',
                             '/train_05_0/train_images/train_05/', '/train_05_1/train_images/train_05/',
                             '/train_05_2/train_images/train_05/', '/train_05_3/train_images/train_05/',
                             '/train_05_4/train_images/train_05/', '/train_05_5/train_images/train_05/',
                             '/train_05_6/train_images/train_05/',
                             '/train_06_0/train_images/train_06/', '/train_06_1/train_images/train_06/',
                             '/train_06_2/train_images/train_06/', '/train_06_3/train_images/train_06/',
                             '/train_06_4/train_images/train_06/', '/train_06_5/train_images/train_06/',
                             '/train_06_6/train_images/train_06/',
                             '/train_07_0/train_images/train_07/', '/train_07_1/train_images/train_07/',
                             '/train_07_2/train_images/train_07/', '/train_07_3/train_images/train_07/',
                             '/train_07_4/train_images/train_07/', '/train_07_5/train_images/train_07/',
                             '/train_07_6/train_images/train_07/',
                             '/train_08_0/train_images/train_08/', '/train_08_1/train_images/train_08/',
                             '/train_08_2/train_images/train_08/', '/train_08_3/train_images/train_08/',
                             '/train_08_4/train_images/train_08/']
        
        img_prefix_list = ['/train_images/train'+str(i) for i in range(parts_per_dataset)]

        import numpy as np
        np.random.seed(random_seed)
        np.random.shuffle(unzip_file_names)
        np.random.seed(random_seed)
        np.random.shuffle(unzip_prefix_list)

        if mode == 'generating':
            start_num = generating_start_num
        else: 
            start_num = already_trained_iters

        unzip_thread = threading.Thread(target=general_processing.load_and_unzip_train_data, 
                                        args=(file_prefix, [unzip_file_names[start_num]]))
        unzip_thread.daemon = True
        unzip_thread.start()

    elif mode == 'test':
        file_prefix = main_path
        unzip_file_names = ['submit_test.zip']

        unzip_thread = threading.Thread(target=general_processing.load_and_unzip_train_data, 
                                        args=(file_prefix, [unzip_file_names[0]]))
        unzip_thread.daemon = True
        unzip_thread.start()
    
    general_processing.install_packages(['torch==1.1.0', 'pytest-runner'])
    
    """
    # threads for unzip
    unzip_threads = []
    unzip_file_names = ['train_00']
    for unzip_file_name in unzip_file_names:
        unzip_threads.append(threading.Thread(target=general_processing.quiet_system, 
                                                args=('unzip -d /cos_person/275/1745/object_detection/train_images/ '+main_path+unzip_file_name,)))
        unzip_threads[-1].daemon = True
        unzip_threads[-1].start()
    unzip_file_names = ['validation']
    for unzip_file_name in unzip_file_names:
        unzip_threads.append(threading.Thread(target=general_processing.quiet_system, 
                                                args=('unzip -d /cos_person/275/1745/object_detection/val_images/ '+main_path+unzip_file_name,)))
        unzip_threads[-1].daemon = True
        unzip_threads[-1].start()
    """

    os.system("""cd /cos_person/275/1745/mmdetection/mmdetection;
                    sh compile.sh;   
                    pip install -e .""")
    """
    # wait for unzip
    for unzip_thread in unzip_threads:
        unzip_thread.join()
    """
    config_file = 'dcn_best.py'
    output_path = "/cos_person/275/1745/object_detection/output/"
    checkpoint_file = output_path + 'epoch_1.pth'

    """
    def multiple_data_preprocessing(prefix_list):
        '''serially preprocess datasets'''

        for prefix in prefix_list:
            preprocessor = general_processing.data_preprocessor(prefix)
            preprocessor.generate_csv()
    """

    # start
    if (mode=='training') or (mode=='generating'):
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        import pickle
        
        unzip_thread.join()

        if (total_training_iters==0) or (mode=='generating'):
            total_iters = len(unzip_file_names)
        else:
            total_iters = total_training_iters

        for i in range(start_num, total_iters):
            
            print("iter:"+str(i))
            # set current dataset
            path_list = general_processing.get_file_paths_from_folder(unzip_prefix_list[i][:-1])

            for j in range(parts_per_dataset):
                raw_dataset_path_name_list = sampling_record_path + 'raw_dataset' +str(i) + '_' + str(j) + '.pl'
                os.system('mkdir -p '+img_prefix_list[j])
                try:
                    with open(raw_dataset_path_name_list, 'rb') as f:
                        small_path_list = pickle.load(f)
                except:
                    if j < (parts_per_dataset-1):
                        small_path_list = path_list[(len(path_list)//parts_per_dataset)*j:(len(path_list)//parts_per_dataset)*(j+1)]
                    else:
                        small_path_list = path_list[(len(path_list)//parts_per_dataset)*j:]
                    with open(raw_dataset_path_name_list, 'wb') as f:
                        pickle.dump(small_path_list, f)

                for k in range(len(small_path_list)):
                    os.system('cp -r '+small_path_list[k]+' '+img_prefix_list[j])

                # preprocess current dataset
                sampling_csv_path_name = sampling_record_path + 'ann' +str(i) + '_' + str(j) + '.csv'
                garbage_pl_path_name = sampling_record_path + 'garbage' +str(i) + '_' + str(j) + '.pl'

                general_processing.data_preprocessor(img_prefix_list[j], sampling_csv_path_name, garbage_pl_path_name).generate_csv()

                # in fact, past_iteration is unnecessary
                if mode == 'training':
                    os.system('python '+mmdetection_path+'tools/train.py '+main_path+'code/'+config_file+\
                        ' --gpus 1 --work_dir '+output_path+' --past_iteration '+str((i*parts_per_dataset)+j)+\
                        ' --dataset '+img_prefix_list[j]+' --ann_csv '+sampling_csv_path_name)
                    #if i>quiet_training_iters and j==0:
                    #os.system('python '+main_path+'code/output_test.py --iter_num '+ str(i*parts_per_dataset+j))
                os.system('rm -rf '+img_prefix_list[j])
            if i < (total_iters-1):
                general_processing.load_and_unzip_train_data(file_prefix, [unzip_file_names[i+1]])
        os.system('python '+main_path+'code/output_test.py --iter_num '+ str(i*parts_per_dataset+j))
    elif mode == 'debug_test':
        #os.system('python '+mmdetection_path+'tools/test.py '+main_path+'code/'+config_file+' '+checkpoint_file+' --out '+output_path+'test.pkl')  
        os.system('python '+main_path+'code/output_test.py --iter_num -1')
    elif mode == 'test':
        os.system('python '+main_path+'code/test_submit_output.py '+main_path+'code/'+config_file+' '+checkpoint_file)
