import tensorflow as tf
import numpy as np
import pandas as pd

from detector import Detector
import util
import os
import ipdb

images_path = '../data/images/'
annotation_path = '../data/annotations/'
real_images_path = '../data/real_data/'

syn_real_kitchen_path = '../data/syn_real_kitchen/'
syn_trainset_path = '../data/syn_real_kitchen/syn_train.pickle'
real_trainset_path = '../data/syn_real_kitchen/real_train.pickle'
real_testset_path = '../data/syn_real_kitchen/test.pickle'
syn_label_dict_path = '../data/syn_real_kitchen/syn_label_dict.pickle'

if not os.path.exists( syn_trainset_path ):
    if not os.path.exists( syn_real_kitchen_path ):
        os.makedirs( syn_real_kitchen_path )
    image_dir_list = os.listdir(images_path )
    annotation_index = [int(x.split('_')[0]) for x in image_dir_list]
    names_labels_dict = util.names_labels_dictionary('../data/labels.txt')
    label_pairs = [util.get_labels_names_array(ind,annotation_path,names_labels_dict) for ind in annotation_index]
    labels, label_names = list(zip(*label_pairs))
    n_labels = len(names_labels_dict)-1
    image_paths_train = [os.path.join(images_path,file) for file in image_dir_list]
    trainset = pd.DataFrame({'image_path': image_paths_train})
    trainset = trainset[ trainset['image_path'].map( lambda x: x.endswith('.jpg'))]
    trainset['label'] = labels
    trainset['label_names'] = label_names
    label_dict = pd.Series( [index for index in names_labels_dict.values()], index=[names for names in names_labels_dict.keys()] )
    label_dict.to_pickle(syn_label_dict_path)
    trainset.to_pickle(syn_trainset_path)

if not os.path.exists(real_trainset_path) or not os.path.exists(real_testset_path):
    real_image_folder_list = os.listdir(real_images_path)
    names_labels_dict = util.names_labels_dictionary('../data/labels.txt')
    image_paths_train = []
    image_paths_test = []
    labels_train = []
    label_names_train = []
    labels_test = []
    label_names_test = []
    for cnt,scene_folder in enumerate(real_image_folder_list):
        scene_dir_list = os.listdir(os.path.join(real_images_path,scene_folder))
        image_dir = os.path.join(real_images_path,scene_folder,scene_dir_list[0])
        annotation_dir = os.path.join(real_images_path,scene_folder,scene_dir_list[1])
        image_dir_list = os.listdir(image_dir)
        annotation_index = [x.split('.')[0] for x in image_dir_list]
        label_pairs = [util.get_labels_names_array(ind,annotation_dir,names_labels_dict) for ind in annotation_index]
        labels, label_names = list(zip(*label_pairs))
        image_paths = [os.path.join(image_dir,file) for file in image_dir_list]
        #print(len(labels),len(label_names),len(image_paths))
        if cnt<1:
            image_paths_train += image_paths
            labels_train += labels
            label_names_train += label_names
        else:
            image_paths_test += image_paths
            labels_test += labels
            label_names_test += label_names
    trainset = pd.DataFrame({'image_path': image_paths_train})
    trainset['label'] = labels_train
    trainset['label_names'] = label_names_train
    trainset.to_pickle(real_trainset_path)
    
    testset = pd.DataFrame({'image_path': image_paths_test})
    testset['label'] = labels_test
    testset['label_names'] = label_names_test
    testset.to_pickle(real_testset_path)