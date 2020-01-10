import detection_inference
import tensorflow as tf
import general_processing
import numpy as np

model_file = 'D:\\Backup\\Documents\\Visual Studio 2015\\faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12\\frozen_inference_graph.pb'

def output_list_by_pbfile(data_folder, model_file=model_file):
    """output a list of predictions on data in data_path"""

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # get tensors
    input_tensor = tf.placeholder(tf.uint8, [1, None, None, 3])
    box_tensor, score_tensor, label_tensor = detection_inference.build_inference_graph(input_tensor, model_file)

    # get data paths
    data_paths = general_processing.get_file_paths_from_folder(data_folder)
 
    output_list = []
    batch_generator = general_processing.batch_generator(data_paths, None, None, None, 1)
    image_data_list = batch_generator.get_specific_batch(data_paths)
    for image_data in image_data_list:
        output_list.append(sess.run([box_tensor, score_tensor, label_tensor], {input_tensor:np.expand_dims(image_data*255, 0)}))

    return output_list

#def output_csv_by_pbfile(data_folder, model_file=model_file):
    """output a csv file of predictions on data in data_path"""

    
