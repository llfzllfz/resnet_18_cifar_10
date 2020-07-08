import tensorflow as tf
import config

class Network:
    def __init__(self):
        self.batch_size = config.batch_size
        self.x = tf.placeholder('float',[self.batch_size,32,32,3])
        self.y = tf.placeholder('float',[self.batch_size,10])
    
    def conv2d(self,inputs,shapes,size,strides,relus = 1):
        w = tf.Variable(tf.random_normal(shape = shapes,stddev = 0.01), dtype = tf.float32)
        b = tf.Variable(tf.random_normal(shape = size), dtype = tf.float32)
        if relus == 1:
            outputs = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputs,filter = w,strides = strides,padding = 'SAME'), b))
        else:
            outputs = (tf.nn.bias_add(tf.nn.conv2d(inputs,filter = w,strides = strides,padding = 'SAME'), b))
        return outputs
    
    def work(self):
        # conv1
        conv1 = self.conv2d(self.x,[7,7,3,64],[64],[1,2,2,1])
        pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
        # conv2_x
        conv2_1 = self.conv2d(pool1,[3,3,64,64],[64],[1,1,1,1])
        conv2_2 = self.conv2d(conv2_1,[3,3,64,64],[64],[1,1,1,1])
        conv2_3 = pool1 + conv2_2
        conv2_4 = self.conv2d(conv2_3,[3,3,64,64],[64],[1,1,1,1])
        conv2_5 = self.conv2d(conv2_4,[3,3,64,64],[64],[1,1,1,1])
        conv2 = conv2_3 + conv2_5
        # conv3_x
        conv3_1 = self.conv2d(conv2,[3,3,64,128],[128],[1,2,2,1])
        conv3_1_1 = self.conv2d(conv2,[1,1,64,128],[128],[1,2,2,1],0)
        conv3_2 = self.conv2d(conv3_1,[3,3,128,128],[128],[1,1,1,1])
        conv3_3 = conv3_1_1 + conv3_2
        conv3_4 = self.conv2d(conv3_3,[3,3,128,128],[128],[1,1,1,1])
        conv3_5 = self.conv2d(conv3_4,[3,3,128,128],[128],[1,1,1,1])
        conv3 = conv3_3 + conv3_5
        
        # conv4_x
        conv4_1 = self.conv2d(conv3,[3,3,128,256],[256],[1,2,2,1])
        conv4_1_1 = self.conv2d(conv3,[1,1,128,256],[256],[1,2,2,1],0)
        conv4_2 = self.conv2d(conv4_1,[3,3,256,256],[256],[1,1,1,1])
        conv4_3 = conv4_1_1 + conv4_2
        conv4_4 = self.conv2d(conv4_3,[3,3,256,256],[256],[1,1,1,1])
        conv4_5 = self.conv2d(conv4_4,[3,3,256,256],[256],[1,1,1,1])
        conv4 = conv4_3 + conv4_5
        # conv5_x
        conv5_1 = self.conv2d(conv4,[3,3,256,512],[512],[1,2,2,1])
        conv5_1_1 = self.conv2d(conv4,[1,1,256,512],[512],[1,2,2,1])
        conv5_2 = self.conv2d(conv5_1,[3,3,512,512],[512],[1,1,1,1])
        conv5_3 = conv5_1_1 + conv5_2
        conv5_4 = self.conv2d(conv5_3,[3,3,512,512],[512],[1,1,1,1])
        conv5_5 = self.conv2d(conv5_4,[3,3,512,512],[512],[1,1,1,1])
        conv5 = conv5_3 + conv5_5
        
        # avgpool
        pool2 = tf.nn.avg_pool(conv5, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
        # dense
        output = self.conv2d(pool2,[1,1,512,10],[10],[1,1,1,1],0)
        return output
    
    
