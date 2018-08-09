import time
import tensorflow as tf
import face_generateds
import face_forward
import face_backword
TIME_INTERVAL_SECS=5 
TEST_NUM = 2190
def test():
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,face_forward.INPUT_NODE])
        y_=tf.placeholder(tf.float32,[None,face_forward.OUTPUT_NODE])
        y=face_forward.forward(x,None)
      
        ema = tf.train.ExponentialMovingAverage(face_backword.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)

        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        img_batch, label_batch = face_generateds.get_tfRecode(TEST_NUM,isTrain=False)
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(face_backword.MODEL_SAVE_PATH)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    xs, ys = sess.run([img_batch,label_batch])
                    accuracy_score=sess.run(accuracy,feed_dict={x:xs,y_:ys})
                    print("After %s traing steps (s),test accuracy = % g"%(global_step,accuracy_score))
                else:
                    print("Without checkpoint")
            time.sleep(TIME_INTERVAL_SECS)
def main():
    test()

if __name__ == "__main__":
    main()
