# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data
import model
import Image
testbatch=100
N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 30   #每次是用１６个大小的图片　作为一个批次
CAPACITY = 2000
MAX_STEP = 10000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001





#這是按批次測試放方法．
def get_one_image():
    # you need to change the directories to yours.
    train_dir = '/home/maqunfei/PycharmProjects/catvsdog/data/train/train'
    logs_train_dir = '/home/maqunfei/PycharmProjects/catvsdog/logs/train/train'
    #调用得到解析出来的数据
    train, train_label = input_data.getfiles(train_dir)
    #获取　批次数据的批操作
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)

    #是用模型进行训练得到预测结果列表
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    #train_logits = tf.nn.softmax(train_logits)
    #print 'train_logits:'+train_logits
    train_loss = model.losses(train_logits, train_label_batch)
    #train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
    logs_train_dir = '/home/maqunfei/PycharmProjects/catvsdog/logs/train/train'
    saver = tf.train.Saver()
    # with tf.Session() as sess:
    #
    #     print("Reading checkpoints...")
    #     # get_checkpoint_state函数会从文件中找到最新的模型文件名(注意时最新，你可能新跑着试着玩可能被利用上了,所以别随便乱跑)
    #     ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #         print('Loading success, global_step is %s' % global_step)
    #     else:
    #         print('No checkpoint file found')

    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print 'global_step:',global_step
    #sess.run(tf.global_variables_initializer())#絕對不能家
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            # 这是进行每次训练的步骤　　　每次可以得到准确率等值     這裏執行train_op   每次的操作都會調用取數據操作
            tra_acc = sess.run(train__acc)
            # print 'tra_acc:', logits
            # print 'tra_acc:', labels
            print 'tra_acc:',tra_acc
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

get_one_image()




# def get_one_image():
#     train_dir = '/home/maqunfei/PycharmProjects/catvsdog/data/train/train'
#     logs_train_dir = '/home/maqunfei/PycharmProjects/catvsdog/logs/train/train'
#     #调用得到解析出来的数据
#     train, train_label = input_data.getfiles(train_dir)
#     #从训练图片中随即取一张图片
#     n=len(train)
#     ind=np.random.randint(0,n)
#     img_dir=train[ind]
#     image=Image.open(img_dir)
#     #plt.imshow(image)
#     #plt.show()
#     #对图片格式进行处理
#     image=image.resize([208,208])
#     image=np.array(image)
#     print 'true label:',train_label[ind]
#     return image,train_label[ind]
#
# evaluate_one_image()
# def evaluate_one_image():
#     train_dir='/home/maqunfei/PycharmProjects/catvsdog/data/train/train'
#     train,train_label=input_data.getfiles(train_dir)
#     true_number=0#記錄分類正確的樣本數
#     #現在從樣本幾中找１００個圖片進行測試
#     for i in range(0,100):
#         image_array,label_array=get_one_image()
#         with tf.Graph().as_default():
#             BATCH_SIZE = 1
#             N_CLASSES = 2
#
#             image = tf.cast(image_array, tf.float32)
#             image = tf.image.per_image_standardization(image)
#             image = tf.reshape(image, [1, 208, 208, 3])
#
#             logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#
#             logit = tf.nn.softmax(logit)
#
#             x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#
#             # you need to change the directories to yours.
#             logs_train_dir = '/home/maqunfei/PycharmProjects/catvsdog/logs/train/train'
#
#             saver = tf.train.Saver()
#
#             with tf.Session() as sess:
#
#                 print("Reading checkpoints...")
#                 #get_checkpoint_state函数会从文件中找到最新的模型文件名(注意时最新，你可能新跑着试着玩可能被利用上了,所以别随便乱跑)
#                 ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#                 if ckpt and ckpt.model_checkpoint_path:
#                     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                     saver.restore(sess, ckpt.model_checkpoint_path)
#                     print('Loading success, global_step is %s' % global_step)
#                 else:
#                     print('No checkpoint file found')
#
#                 prediction = sess.run(logit, feed_dict={x: image_array})
#                 print 'prediction:',prediction
#                 max_index = np.argmax(prediction)
#                 if max_index == 0:
#                     print('This is a cat with possibility %.6f' % prediction[:, 0])
#                 else:
#                     print('This is a dog with possibility %.6f' % prediction[:, 1])
#                 if max_index==label_array:
#                     true_number+=1
#     print 'true_number:',true_number
#     print 'the '
# evaluate_one_image()



