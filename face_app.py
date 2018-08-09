from PIL import Image
import tensorflow as tf
import face_backword
import face_forward
import numpy as np
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x=tf.placeholder(tf.float32,[None,face_forward.INPUT_NODE])
        y=face_forward.forward(x,None)
        preValue=tf.argmax(y,1)

        variable_averages=tf.train.ExponentialMovingAverage(face_backword.MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver= tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt =tf.train.get_checkpoint_state(face_backword.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)

                preValue=sess.run(preValue,feed_dict={x:testPicArr})
                return preValue
            else:
                print " No checkPoint file found"
                return -1
def pre_pic(picname):
    img=Image.open(picname)
    reIm=img.resize((28,28),Image.ANTIALIAS)
    im_arr=np.array(reIm.convert('L'))
    nu_arr=im_arr.reshape([1,784])
    nu_arr=nu_arr.astype(np.float32)
    img_ready=np.multiply(nu_arr,1.0/255.0)
    return img_ready
def application():
    testNum=input("Input the number of the test pictures:")
    for i in range(testNum):
        testPic = raw_input("the path of the picture:")
        testPicArr=pre_pic(testPic)
        preValue=restore_model(testPicArr)
        print "The number is",preValue

if __name__ == "__main__":
    application()
