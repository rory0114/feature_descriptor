import tensorflow as tf

#%%
def inference(images):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    
    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights', 
                                  shape = [5,5,1,32],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
    
    #pool1 and norm1   
    with tf.variable_scope('pooling1_lrn',reuse=tf.AUTO_REUSE) as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='VALID', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')
    
    #conv2
    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,32,64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn',reuse=tf.AUTO_REUSE) as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,2,2,1], strides=[1,2,2,1],
                               padding='SAME',name='pooling2')
    
    #conv3, shape = [kernel size, kernel size, channels, kernel numbers]
    
    with tf.variable_scope('conv3',reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3,3,64, 96],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[96],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1,1,1,1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name= scope.name)
    
    #pool4 and norm4   
    with tf.variable_scope('pooling3_lrn',reuse=tf.AUTO_REUSE) as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='VALID', name='pooling3')
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm3')
        #conv3, shape = [kernel size, kernel size, channels, kernel numbers]
    
    with tf.variable_scope('conv4',reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights', 
                                  shape = [5,5,96, 128],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm3, weights, strides=[1,1,1,1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name= scope.name)
    
    #pool4 and norm4   
    with tf.variable_scope('pooling4_lrn',reuse=tf.AUTO_REUSE) as scope:
        pool4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='VALID', name='pooling3')
        norm4 = tf.nn.lrn(pool4, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm4')

    return norm4


#%%
def losses(Y1,Y2, labels,k):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
    
    YY1=tf.reshape(Y1,[-1,128])
    YY2=tf.reshape(Y2,[-1,128])
    labels = tf.reshape(labels,[-1])
    labels = tf.cast(labels,tf.float32)
    
    with tf.variable_scope('loss',reuse = tf.AUTO_REUSE) as scope:
        #cost =labels*tf.losses.mean_squared_error(YY1,YY2)+(1-labels)*tf.where(tf.greater((4-tf.losses.mean_squared_error(YY1,YY2)), labels), (4-tf.losses.mean_squared_error(YY1,YY2))*(1-labels), labels)
        square=tf.reduce_sum(tf.square(YY1-YY2),1)
        print(square,"square")
        cost1 = labels*square+(1-labels)*tf.where((4-square)>0,4-square,square*0)
        costt = tf.reduce_mean(cost1)
        cost2 = tf.reshape(cost1,[-1])
        cost3,_ = tf.nn.top_k(cost2,k)
        cost4 = tf.reshape(cost3,[-1,1])
        cost = tf.reduce_mean(cost4)
        tf.summary.scalar(scope.name+'/loss', cost)
        tf.summary.scalar(scope.name+'/full_loss', costt)
    return cost,costt,square,labels

#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
##def evaluation(logits, labels):
##  """Evaluate the quality of the logits at predicting the label.
##  Args:
##    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
##    labels: Labels tensor, int32 - [batch_size], with values in the
##      range [0, NUM_CLASSES).
##  Returns:
##    A scalar int32 tensor with the number of examples (out of batch_size)
##    that were predicted correctly.
##  """
##  with tf.variable_scope('accuracy') as scope:
##      correct = tf.nn.in_top_k(logits, labels, 1)
##      correct = tf.cast(correct, tf.float16)
##      accuracy = tf.reduce_mean(correct)
##      tf.summary.scalar(scope.name+'/accuracy', accuracy)
##  return accuracy

#%%

