def evaluate_test():
 
    # you need to change the directories to yours.
    test_dir = 'E:/学习/毕设/dataset/london_tower/patches'
    
    
    with tf.Graph().as_default():
        BATCH_SIZE = 100

        test= input_data.get_testfiles(test_dir)

        allnum = len(test)

# =============================================================================
        test_batch = input_data.get_testbatch(test
                                                           BATCH_SIZE, 
                                                           CAPACITY)
     
 

        testy = []
        disy = []
        
        for i in range(0,len(test_batch)):
            testy[i] = model.inference( test_batch[i])
            
        
        for i in range(0,len(test_batch)):
            testy[i] = tf.reshape(testy,[-1,128])

        for i in range(0,len(test_batch)):
            disy[i] =  tf.reduce_sum(tf.square(testy[i]-testy[0]),1)

        print(disy)
        
            
##        testy2 = model.inference( testx2_batch)
##        ty1=tf.reshape(testy1,[-1,128])
##        ty2=tf.reshape(testy2,[-1,128])
##        test_label_batch1 = tf.reshape(test_label_batch,[-1])
##        test_label_batch2 = tf.cast(test_label_batch1,tf.int32)
##        hh=tf.reduce_sum(tf.square(ty1-ty2),1)
##        
##        test_loss=tf.reduce_mean(hh,0)
##        
##        test_h= (hh<2)
##        testh=tf.cast(test_h,tf.int32)
##        
##        right = tf.equal(testh,test_label_batch2)
##        rightint = tf.cast(right,tf.int32)
##        rightnum = tf.reduce_sum(rightint)
    # =============================================================================
            # you need to change the directories to yours.
        logs_train_dir = 'C:/log/' 
                       
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            print("Reading checkpoints...")
            
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                
            try:
                for step in np.arange(MAX_STEP):
                    if coord.should_stop():
                            break
                    a= sess.run([rightnum])
                    print("the accuracy is  ",a)

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            
            finally:
                coord.request_stop()
                
            coord.join(threads)
            
