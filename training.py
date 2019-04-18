# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import input_data
import model

# %%

N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 16   #每次是用１６个大小的图片　作为一个批次
CAPACITY = 2000
MAX_STEP = 16000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001


# %%
def run_training():
    # you need to change the directories to yours.
    train_dir = '/home/maqunfei/PycharmProjects/catvsdog/data/train/train'
    logs_train_dir = '/home/maqunfei/PycharmProjects/catvsdog/logs/train/train'
    print 'test   11'
    #调用得到解析出来的数据
    train, train_label = input_data.getfiles(train_dir)
    print 'test   21'
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
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
    #启动所有汇总语句
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    #将所有汇总进行保存
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            #这是进行每次训练的步骤　　　每次可以得到准确率等值     這裏執行train_op   每次的操作都會調用取數據操作
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
            ####################################################每次迭代中的輸出   真实標籤和预测标签的预测
            #print 'true label',sess.run(train_label_batch)# 這地方很巧妙，相當於運行了　train_label_batch節點，返回了train_label_batch所包含的值

            #old_showlogit=sess.run(train_logits)
            #print 'old prediction label：', old_showlogit
            #showlogit = b=[i[1] > i[0] and 1 or 0  for i  in old_showlogit] #進行處理下，第一個大設置爲０　　第二個大設置爲１
            #print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            #print 'prediction label：',showlogit
            #####################################################
            if step % 50 == 0:
                #　　　每隔５０步　　将模型中得到的损失和准确率输出来
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                #每隔５０布进行汇总　并将loss和acc记录　并写入日志文件进行记录
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                #每隔２０００步骤　将模型结构进行保存
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

run_training()
    # %% Evaluate one image
    # when training, comment the following codes.


    # from PIL import Image
    # import matplotlib.pyplot as plt
    #
    # def get_one_image(train):
    #    '''Randomly pick one image from training data
    #    Return: ndarray
    #    '''
    #    n = len(train)
    #    ind = np.random.randint(0, n)
    #    img_dir = train[ind]
    #
    #    image = Image.open(img_dir)
    #    plt.imshow(image)
    #    image = image.resize([208, 208])
    #    image = np.array(image)
    #    return image
    #
    # def evaluate_one_image():
    #    '''Test one image against the saved models and parameters
    #    '''
    #
    #    # you need to change the directories to yours.
    #    train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
    #    train, train_label = input_data.get_files(train_dir)
    #    image_array = get_one_image(train)
    #
    #    with tf.Graph().as_default():
    #        BATCH_SIZE = 1
    #        N_CLASSES = 2
    #
    #        image = tf.cast(image_array, tf.float32)
    #        image = tf.image.per_image_standardization(image)
    #        image = tf.reshape(image, [1, 208, 208, 3])
    #        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
    #
    #        logit = tf.nn.softmax(logit)
    #
    #        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
    #
    #        # you need to change the directories to yours.
    #        logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'
    #
    #        saver = tf.train.Saver()
    #
    #        with tf.Session() as sess:
    #
    #            print("Reading checkpoints...")
    #            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    #            if ckpt and ckpt.model_checkpoint_path:
    #                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #                saver.restore(sess, ckpt.model_checkpoint_path)
    #                print('Loading success, global_step is %s' % global_step)
    #            else:
    #                print('No checkpoint file found')
    #
    #            prediction = sess.run(logit, feed_dict={x: image_array})
    #            max_index = np.argmax(prediction)
    #            if max_index==0:
    #                print('This is a cat with possibility %.6f' %prediction[:, 0])
    #            else:
    #                print('This is a dog with possibility %.6f' %prediction[:, 1])


    # %%