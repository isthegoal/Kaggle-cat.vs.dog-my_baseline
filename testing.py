# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data
import model
import Image
testbatch=100
def get_one_image(train,train_label):
    #从训练图片中随即取一张图片
    n=len(train)
    ind=np.random.randint(0,n)
    img_dir=train[ind]
    image=Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    #对图片格式进行处理
    image=image.resize([208,208])
    image=np.array(image)
    print 'true label:',train_label[ind]
    return image,train_label[ind]

def evaluate_one_image():
    train_dir='/home/maqunfei/PycharmProjects/catvsdog/data/train/train'
    train,train_label=input_data.getfiles(train_dir)
    true_number=0#記錄分類正確的樣本數
    #現在從樣本幾中找１００個圖片進行測試
    for i in range(0,100):
        image_array,label_array=get_one_image(train,train_label)
        with tf.Graph().as_default():
            BATCH_SIZE = 1
            N_CLASSES = 2

            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, [1, 208, 208, 3])

            logit = model.inference(image, BATCH_SIZE, N_CLASSES)

            logit = tf.nn.softmax(logit)

            x = tf.placeholder(tf.float32, shape=[208, 208, 3])

            # you need to change the directories to yours.
            logs_train_dir = '/home/maqunfei/PycharmProjects/catvsdog/logs/train/train'

            saver = tf.train.Saver()

            with tf.Session() as sess:

                print("Reading checkpoints...")
                #get_checkpoint_state函数会从文件中找到最新的模型文件名(注意时最新，你可能新跑着试着玩可能被利用上了,所以别随便乱跑)
                ckpt = tf.train.get_checkpoint_state(logs_train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')

                prediction = sess.run(logit, feed_dict={x: image_array})
                print 'prediction:',prediction
                max_index = np.argmax(prediction)
                if max_index == 0:
                    print('This is a cat with possibility %.6f' % prediction[:, 0])
                else:
                    print('This is a dog with possibility %.6f' % prediction[:, 1])
                if max_index==label_array:
                    true_number+=1
    print 'true_number:',true_number
    print 'the '
evaluate_one_image()



