
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from create_features_and_labels import create_features_and_labels


# In[2]:


N_NODES_HL1 = 500
N_NODES_HL2 = 500
N_NODES_HL3 = 500

N_CLASSES = 10
BATCH_SIZE = 100
HM_EPOCHS = 10

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

hidden_1_layer = {
    'f_fum': N_NODES_HL1,
    'weights': tf.Variable(tf.random_normal([784, N_NODES_HL1])),
    'biases': tf.Variable(tf.random_normal([N_NODES_HL1]))
}

hidden_2_layer = {
    'f_fum': N_NODES_HL2,
    'weights': tf.Variable(tf.random_normal([N_NODES_HL1, N_NODES_HL2])),
    'biases': tf.Variable(tf.random_normal([N_NODES_HL2]))
}

hidden_3_layer = {
    'f_fum': N_NODES_HL3,
    'weights': tf.Variable(tf.random_normal([N_NODES_HL2, N_NODES_HL3])),
    'biases': tf.Variable(tf.random_normal([N_NODES_HL3]))
}

output_layer = {
    'f_fum':None,
    'weights':tf.Variable(tf.random_normal([N_NODES_HL3, N_CLASSES])),
    'biases':tf.Variable(tf.random_normal([N_CLASSES]))
}

# saves the variables for checkpoints
saver = tf.train.Saver();


# In[3]:


def create_nnet_model(data):

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


# In[19]:


def train_neural_network(x, train_x, train_y, test_x, test_y):
    prediction = create_nnet_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(HM_EPOCHS):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+BATCH_SIZE
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i+=BATCH_SIZE

            saver.save(sess, "./tmp/model.ckpt")
            print('Epoch', epoch + 1, 'completed out of', HM_EPOCHS, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


# In[20]:


train_x, train_y = create_features_and_labels("""SELECT * FROM images LIMIT 10000""")


# In[22]:


test_x, test_y = create_features_and_labels("""SELECT *  FROM images ORDER BY random() LIMIT 100""")


# In[23]:


train_neural_network(x, train_x, train_y, test_x, test_y)


# In[ ]:
