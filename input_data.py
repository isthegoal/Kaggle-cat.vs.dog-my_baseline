# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
img_width=208
img_height=208
def getfiles(file_dir):
    cats=[]
    label_cats=[]
    dogs=[]
    label_dogs=[]
    print 'begin'
    #從文件中讀取信息並分開進行標籤和數據的保存
    for file in os.listdir(file_dir):
        name=file.split('.')
        if name[0]=='cat':
            cats.append(file_dir+'/'+file)
            label_cats.append(int(0))
        else:
            dogs.append(file_dir+'/'+file)
            label_dogs.append(int(1))
    print 'the cat num is :',len(cats)
    print 'the dog num is :',len(dogs)
    #此部分可以将　图像和和标签序列序列进行打乱，　　但是注意在这个打乱的过程中，　图像和对应的标签是同序号的．
    img_list=np.hstack((cats,dogs))
    label_list=np.hstack((label_cats,label_dogs))
    temp=np.array([img_list,label_list])
    temp=temp.transpose()
    np.random.shuffle(temp)


    #将图像和标签分别抽取并保存．
    image_list=list(temp[:,0])
    label_list=list(temp[:,1])
    print type(label_list[1])
    label_list=[int(float(i)) for i in label_list]
    return image_list,label_list

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    # 将python.list类型转换成tf能够识别的格式　　　　　用于对图像进行处理
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列　　　　　　slice_input_producer可以创建队列　　　也不知道怎么想得，这里是先取得部分放到队列中，然后再放进去下一个批次到队列　
    input_queue = tf.train.slice_input_producer([image, label],shuffle=True)
    #image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    #将image读出来并进行解码
    image_contents=tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 统一图片大小
    # 视频方法
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 我的方法　　　　　　　　对图像进行剪切处理  　使用resize_images()对图像进行缩放，而不是裁剪，这方法不同与视频中原带的方法是一种缩放，能让结果更准确
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #image = tf.cast(image, tf.float32)   #圖片看上去不正常是因爲這個原因，　轉換浮點型之後圖像看上去會不正常．
    #image = tf.image.per_image_standardization(image)
    #  产生图像的批次　　　提取批次，前面的都是合在一起的　　每个批次将batch_size大小的　image和lable送到网络进行训练，　　队列的大小是　
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,   # 线程
                                              capacity=capacity)
    print 'every   label_batch:',label_batch
    # 这行多余？
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

# # TEST
# BATCH_SIZE = 5
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208
#
# train_dir = "/home/maqunfei/PycharmProjects/catvsdog/data/train/train"
# image_list, label_list = getfiles(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:#创建运行用的回话
#     i = 0
#     #coordinate是用来帮助（运行队列）多个线程同时停止，　起到协调器的作用　　　所以这里多用于判断队列的那个批次进数据是否进完了
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     try:
#         #这里i可以指定循环的次数， i的限制指定了执行多少个BATCHsize步骤　　　　　　　　　
#         while not coord.should_stop() and i < 1:
#             # 这里session.run是运行时每次运行一个批次每个批次展示５张图片，　下次　while循环会执行下一个批次的提取．　　启动一个图运算
#             img, label = sess.run([image_batch, label_batch])
#             for j in np.arange(BATCH_SIZE):
#                  print("label: %d" % label[j])
#                  plt.imshow(img[j, :, :, :])
#                  plt.show()
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print("done!")
#     finally:
#         coord.request_stop()
#     coord.join(threads)


# if __name__=='__main__':
#     image_list,label_list=getfiles('/home/maqunfei/PycharmProjects/catvsdog/data/train/train')


