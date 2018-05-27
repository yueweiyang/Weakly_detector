import tensorflow as tf
import numpy as np
import pandas as pd

from detector import Detector
from util import load_image
import os
import ipdb

weight_path = '../data/caffe_layers_value.pickle'
model_path = '../models/caltech256/'
pretrained_model_path = None #'../models/caltech256/model-0'
n_epochs = 10000
init_learning_rate = 0.01
weight_decay_rate = 0.0005
momentum = 0.9
batch_size = 60

syn_real_kitchen_path = '../data/syn_real_kitchen/'
syn_trainset_path = '../data/syn_real_kitchen/syn_train.pickle'
real_trainset_path = '../data/syn_real_kitchen/real_train.pickle'
real_testset_path = '../data/syn_real_kitchen/test.pickle'
syn_label_dict_path = '../data/syn_real_kitchen/syn_label_dict.pickle'

trainset = pd.read_pickle( syn_trainset_path )
testset  = pd.read_pickle( real_testset_path )
label_dict = pd.read_pickle( syn_label_dict_path )
n_labels = len(label_dict)-1

learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

detector = Detector(weight_path, n_labels)

p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)
loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits = output, labels = labels_tf ))

weights_only = [x for x in tf.trainable_variables() if x.name.endswith('W:0')]
weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
loss_tf += weight_decay

sess = tf.InteractiveSession()
saver = tf.train.Saver( max_to_keep=50 )

optimizer = tf.train.MomentumOptimizer( learning_rate, momentum )
grads_and_vars = optimizer.compute_gradients( loss_tf )
grads_and_vars = [(gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars]
#grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
train_op = optimizer.apply_gradients( grads_and_vars )
tf.initialize_all_variables().run()

if pretrained_model_path:
    print("Pretrained")
    saver.restore(sess, pretrained_model_path)

testset.index  = list(range( len(testset)))
#testset = testset.ix[np.random.permutation( len(testset) )]#[:1000]
#trainset2 = testset[1000:]
#testset = testset[:1000]

#trainset = pd.concat( [trainset, trainset2] )
# We lack the number of training set. Let's use some of the test images

f_log = open('../results/log.caltech256.txt', 'w')

iterations = 0
loss_list = []
for epoch in range(n_epochs):

    trainset.index = list(range( len(trainset)))
    trainset = trainset.ix[ np.random.permutation( len(trainset) )]

    for start, end in zip(
        list(range( 0, len(trainset)+batch_size, batch_size)),
        list(range(batch_size, len(trainset)+batch_size, batch_size))):

        current_data = trainset[start:end]
        current_image_paths = current_data['image_path'].values
        current_images = np.array([load_image(x) for x in current_image_paths])

        good_index = np.array([x is not None for x in current_images])

        current_data = current_data[good_index]
        current_images = np.stack(current_images[good_index])
        current_labels = current_data['label'].values

        _, loss_val, output_val = sess.run(
                [train_op, loss_tf, output],
                feed_dict={
                    learning_rate: init_learning_rate,
                    images_tf: current_images,
                    labels_tf: current_labels
                    })

        loss_list.append( loss_val )

        iterations += 1
        if iterations % 5 == 0:
            print("======================================")
            print("Epoch", epoch, "Iteration", iterations)
            print("Processed", start, '/', len(trainset))

            label_predictions = output_val.argmax(axis=1)
            acc = (label_predictions == current_labels).sum()

            print("Accuracy:", acc, '/', len(current_labels))
            print("Training Loss:", np.mean(loss_list))
            print("\n")
            loss_list = []

    n_correct = 0
    n_data = 0
    for start, end in zip(
            list(range(0, len(testset)+batch_size, batch_size)),
            list(range(batch_size, len(testset)+batch_size, batch_size))
            ):
        current_data = testset[start:end]
        current_image_paths = current_data['image_path'].values
        current_images = np.array([load_image(x) for x in current_image_paths])

        good_index = np.array([x is not None for x in current_images])

        current_data = current_data[good_index]
        current_images = np.stack(current_images[good_index])
        current_labels = current_data['label'].values

        output_vals = sess.run(
                output,
                feed_dict={images_tf:current_images})

        label_predictions = output_vals.argmax(axis=1)
        acc = (label_predictions == current_labels).sum()

        n_correct += acc
        n_data += len(current_data)

    acc_all = n_correct / float(n_data)
    f_log.write('epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n')
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print('epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n')
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    saver.save( sess, os.path.join( model_path, 'model'), global_step=epoch)

    init_learning_rate *= 0.99




