import tensorflow as tf
import network
import config
import data
import os
tf.reset_default_graph()

def train(new_train = 1):
    networks = network.Network()
    datas = data.cifar_data()
    output = networks.work()
    predicts = tf.nn.softmax(output)
    #print(predicts.shape)
    predicts = tf.reshape(predicts,(config.batch_size,10))
    predicts = tf.argmax(predicts, axis = 1)
    actual_y = tf.argmax(networks.y, axis = 1)
    #print(networks.y.shape)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts,actual_y), dtype = tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = networks.y))
    opt = tf.train.AdamOptimizer(learning_rate = 0.0001)
    train_step = opt.minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        if new_train == 0:
            saver.restore(sess,config.model_path + config.model_name)
        print('Start training...')
        for epoh in range(config.epoh):
            all_loss = 0
            all_acc = 0
            for steps in range(500):
                trainX,trainY = datas.next_batch()
                #print(trainX.shape,trainY.shape)
                _, losses, acc= sess.run([train_step, loss, accuracy],
                                   feed_dict = {
                                       networks.x:trainX,
                                       networks.y:trainY})
                all_loss = all_loss + losses
                all_acc = all_acc + acc
                print('\repoh:{}, step:{}, loss: {}, acc:{}'.format(epoh,steps,all_loss / (steps+1), all_acc / (steps+1)),end="")
            print('\n')
            saver.save(sess,os.path.join(config.model_path,config.model_name))

def test():
    networks = network.Network()
    datas = data.cifar_data()
    output = networks.work()
    predicts = tf.nn.softmax(output)
    #print(predicts.shape)
    predicts = tf.reshape(predicts,(config.batch_size,10))
    predicts = tf.argmax(predicts, axis = 1)
    actual_y = tf.argmax(networks.y, axis = 1)
    #print(networks.y.shape)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts,actual_y), dtype = tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess,config.model_path + config.model_name)
        print('Start testing...')
        all_acc = 0
        for step in range(100):
            X,Y = datas.next_test_batch()
            acc = sess.run([accuracy],
                           feed_dict = {
                               networks.x:X,
                               networks.y:Y})
            all_acc = all_acc + acc[0]
            print('\racc:{}'.format(all_acc / (step+1)), end="")

if __name__ == '__main__':
    #train(0)
    test()