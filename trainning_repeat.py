import os
import numpy as np
import tensorflow as tf
import input_data
import model

#%%

N_CLASSES = 2
BATCH_SIZE = 100
CAPACITY = 2000
MAX_STEP = 1000000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001


#%%
def run_training():
#    X1 = tf.placeholder(tf.float32, shape=(None,64,64,1),name="X1")
#    X2 = tf.placeholder(tf.float32, shape=(None,64,64,1),name="X2")
#    Y = tf.placeholder(tf.float32, shape=(None,1),name="Y")
    # you need to change the directories to yours.
    train_dir1 = 'E:/学习/毕设/dataset/london_tower/patches'
    train_dir2= 'E:/学习/毕设/dataset/london_tower/patches'
    test_dir = 'E:/学习/毕设/dataset/nyc_library/test'
    logs_train_dir = 'E:/log'
    
    train, train_label = input_data.get_files(train_dir1,train_dir2)
#    test1,test2,test_label = input_data.test2data(test_dir)

    
    
    trainx1_batch,trainx2_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          BATCH_SIZE, 
                                                          CAPACITY)
##    testx1_batch,testx2_batch, test_label_batch = input_data.get_batch(test,
##                                                          test_label,
##                                                          BATCH_SIZE, 
##                                                          CAPACITY)
    
    
    
    Y1 = model.inference(trainx1_batch)
    Y2 = model.inference(trainx2_batch)
#    testy1 = model.inference( test1)
#    testy2 = model.inference( test2)
    #square=tf.reduce_sum(tf.square(testy1-testy2),1)
#    test_loss=tf.reduce_mean(square)
#    
#    test_h= (square<2)
#    testh=tf.cast(test_h,tf.int32)
#    m=(testh==test_label)
#    mint=tf.cast(m,tf.int32)
#    h=tf.reduce_sum(mint)
#    print(type(h))
#    accuracy = h
    
    train_loss,trainlosss,sq,labell = model.losses(Y1,Y2, train_label_batch,16)   
    t_h= (sq<2)
    th=tf.cast(t_h,tf.float32)
        
    right = tf.equal(th,labell)
    rightint = tf.cast(right,tf.int32)
    rightnum = tf.reduce_sum(rightint)
    tf.summary.scalar('/right_num_per_batch', rightnum)    
    train_op = model.trainning(train_loss, learning_rate)
    #train__acc = model.evaluation(train_logits, train_label_batch)
       
    summary_op = tf.summary.merge_all()
   
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
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
            _, tra_loss,traloss,sunright = sess.run([train_op, train_loss,trainlosss,rightnum])
               
            if step % 2 == 0:
                print('Step %d, train loss = %.5f full loss = %.5f,have %f  right' %(step, tra_loss,traloss,sunright))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
             
#            if step % 50 == 0:
#                test_loss,accuracy = sess.run([test_loss,accuracy])
#                print('Step %d, test loss = %.5f,accuracy = %.5f' %(step, test_loss,accuracy))
        
            
                
            
            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()





from PIL import Image
import matplotlib.pyplot as plt


def evaluate_test():
 
    # you need to change the directories to yours.
    test_dir = 'E:/学习/毕设/dataset/nyc_library/test'
    
    
    with tf.Graph().as_default():
        BATCH_SIZE = 100

        test,test_label = input_data.get_files(test_dir)

        allnum = len(test)

# =============================================================================
        testx1_batch,testx2_batch, test_label_batch = input_data.get_batch(test,
                                                           test_label,
                                                           BATCH_SIZE, 
                                                           CAPACITY)
     
 

            
        testy1 = model.inference( testx1_batch)
        testy2 = model.inference( testx2_batch)
        ty1=tf.reshape(testy1,[-1,128])
        ty2=tf.reshape(testy2,[-1,128])
        test_label_batch1 = tf.reshape(test_label_batch,[-1])
        test_label_batch2 = tf.cast(test_label_batch1,tf.int32)
        hh=tf.reduce_sum(tf.square(ty1-ty2),1)
        
        test_loss=tf.reduce_mean(hh,0)
        
        test_h= (hh<2)
        testh=tf.cast(test_h,tf.int32)
        
        right = tf.equal(testh,test_label_batch2)
        rightint = tf.cast(right,tf.int32)
        rightnum = tf.reduce_sum(rightint)
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
            
            
if __name__=="__main__":
    evaluate_test()



