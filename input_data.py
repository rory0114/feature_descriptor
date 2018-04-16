import cv2
import os
import numpy as np
import tensorflow as tf

def test_image2data(path):
    image_contents = tf.read_file(path)
    image = tf.image.decode_jpeg(image_contents, channels=1)
    image = tf.reshape(image,[64,640,1])
    print(type(image))
    print(image.shape)
    data = np.split(image, 10, axis=1)
    return data

def raw_image2data(path):
    image_contents = tf.read_file(path)
    image = tf.image.decode_jpeg(image_contents, channels=1)
    image = tf.reshape(image,[64,128,1])
    print(type(image))
    print(image.shape)
    data1 = image[:,:64,:]
    data2 = image[:,64:,:]
    print(type(data1))
    print(data1.shape)

    return data1,data2
def test2data(path):
    idd = 0
    path1 = path+"/match/"
    path2 = path+"/notmatch/"
    
    data1 = np.empty([0,64*64])
    data2 = np.empty([0,64*64])

    label = np.empty([0,1])

 
    for file in os.listdir(path1):
        
        temp = cv2.imdecode(np.fromfile(path1+file, dtype=np.uint8), 0)

        temp1=temp[:,:64]
        temp2=temp[:,64:]
        temp1=temp1.reshape((1,64*64))
        temp2=temp1.reshape((1,64*64))
       
        data1=np.vstack((data1,temp1))
        data2=np.vstack((data2,temp2))
            

        label = np.vstack((label,[[1]]))


    

    for file in os.listdir(path2):
       
        temp = cv2.imdecode(np.fromfile(path2+file, dtype=np.uint8), 0)
    
        temp1=temp[:,:64]
        temp2=temp[:,64:]
        temp1 = temp1.reshape((1,64*64))
        temp2 = temp2.reshape((1,64*64))
        data1=np.vstack((data1,temp1))
        data2=np.vstack((data2,temp2))
        label = np.vstack((label,[[0]]))

    
  

    return data1,data2,label

def get_files(file_dirs):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    image_list = []
    label_list = []
    
    for file_dir in file_dirs:
        print('There are %d matches\n' %(len(image_list)))
        matchfile = file_dir+"/match/"
        notmatchfile = file_dir+"/notmatch/"
        for file in os.listdir(matchfile):
            image_list.append(matchfile+file)
            label_list.append(1)
            
       
        for file in os.listdir(notmatchfile):
            image_list.append(notmatchfile+file)
            label_list.append(0)
        
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
        
    return image_list, label_list
def get_testfiles(file_dirs):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    image_list = []
    
    for file_dir in file_dirs:
        
        file = file_dir+"/test/"
        
        for file in os.listdir(file):
            image_list.append(matchfile+file)

        
    temp = np.array(image_list)
    temp = temp.transpose()
    np.random.shuffle(temp)

    
    image_list = list(temp)
    
    
        
    return image_list

def get_batch(image, label, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    print(type(image),type(label),"1")
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    print(type(image),type(label),"2")

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image = input_queue[0] 
    print(type(image),type(label),image.shape,"3")
    data1,data2 = raw_image2data(image)
    data1 = tf.cast(data1,tf.float32)
    data2 = tf.cast(data2,tf.float32)
    
    ######################################
    # data argumentation should go to here
    ######################################
    data1_batch,data2_batch, label_batch = tf.train.batch([data1,data2, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#   image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    
    data1_batch = tf.cast(data1_batch, tf.float32)
    data2_batch = tf.cast(data2_batch, tf.float32)
    data1_batch=data1_batch/256
    data2_batch=data2_batch/256
    label_batch = tf.cast(label_batch, tf.int32)
    return data1_batch, data2_batch, label_batch

def get_testbatch(image, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    
    

    # make an input queue
    input_queue = tf.train.slice_input_producer([image])
    
    
    image = input_queue[0] 
    print(type(image),type(label),image.shape,"3")
    data = test_image2data(image)
    for eachdata in data:
        eachdata = tf.cast(eachdataset,tf.float32)

   

    
    ######################################
    # data argumentation should go to here
    ######################################
    data_batch = tf.train.batch(data,
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#   image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    for eachdata in data_batch:
        eachdata = eachdata/256
    
    return data_batch
