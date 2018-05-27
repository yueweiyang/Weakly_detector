import skimage.io
import skimage.transform
import ipdb
import os
from xml.etree import cElementTree as ET
import numpy as np

def load_image( path ):
    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    img /= 255.

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [224,224] )
    return resized_img

def names_labels_dictionary(label_file_path):
    dictionary = {}
    with open(label_file_path) as f:
        for line in f:
           (val, key) = line.split()
           dictionary[key] = int(val)
    return dictionary

def get_labels_names_array(annotation_index, annotation_dir, names_labels_dict):
    n_labels = len(names_labels_dict)-1

    file_path = os.path.join(annotation_dir,'{}.xml'.format(annotation_index))
    tree = ET.parse(file_path)
    root = tree.getroot()
    labels = np.zeros(n_labels,dtype=int)
    bndboxes = []
    names = []
    for object in list(root):
        if object.find('name') is not None:
            name = object.find('name').text
            if names_labels_dict.get(name) is not None:
                labels[names_labels_dict.get(name)-1] = 1 #can be modify to tell number of each label
                names.append(name)
                box = object.find('bndbox')
                box_pos = [int(x.text) for x in box]
                bndboxes.append(box_pos)
    return labels,np.unique(names)

 