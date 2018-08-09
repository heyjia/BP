from PIL import Image
import tensorflow as tf
import numpy as np
import os

image_train_path="./train/"
label_train_path="./train/train.txt"
tfRecode_train="./data/mnist_train.tfrecords"
image_test_path="./test/"
label_test_path="./test/test.txt"
tfRecode_test="./data/mnist_test.tfrecords"

data_path="./data"
resize_height = 28
resize_width = 28

def write_tfRecode(tfRecodeName,image_path,label_path):
    writer = tf.python_io.TFRecordWriter(tfRecodeName)
    num_pic = 0
    f=open(label_path,"r")
    contents=f.readlines()
    for content in contents:
        value = content.split(" ")
        img_path = image_path + value[0]
        img = Image.open(img_path)
        img = img.convert("L")
        img_raw = img.tobytes()
        labels = [0] * 8
        # labels[???]=1 
        labels[int(value[1])] = 1
        example = tf.train.Example(features=tf.train.Features(feature={
            "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label":tf.train.Feature(int64_list=tf.train.Int64List(value=labels))}))
        writer.write(example.SerializeToString())
        num_pic +=1
        print("the number of picture ",num_pic)
    writer.close()
    print("writer tfrecord successful!")
        
    
def generate_tfRecode():
    isExists=os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("The data directory was created successfully")
    else:
        print("The directory already exists")
    write_tfRecode(tfRecode_train,image_train_path,label_train_path)
    write_tfRecode(tfRecode_test,image_test_path,label_test_path)
def read_tfRecord(tfRecode_path):
    filename_queue = tf.train.string_input_producer([tfRecode_path])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
        "label": tf.FixedLenFeature([8], tf.int64),
        "img_raw": tf.FixedLenFeature([], tf.string)
        })
    img = tf.decode_raw(features["img_raw"], tf.uint8)
    img.set_shape([784])
    img = tf.cast(img,tf.float32)*(1./255)
    label = tf.cast(features["label"],tf.float32)
    return img,label
def get_tfRecode(num,isTrain):
    if isTrain:
        tfRecode_path=tfRecode_train
    else:
        tfRecode_path=tfRecode_test
    img,label = read_tfRecord(tfRecode_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size = num , num_threads = 2, capacity = 500, min_after_dequeue = 300)
    return img_batch, label_batch
def main():
    generate_tfRecode()


if __name__ == "__main__":
    main()
