import numpy as np
import tensorflow as tf
from create_features_and_labels import create_features_and_labels


class NeuralNetModel:
    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self):

        self.N_NODES_HL1 = 500
        self.N_NODES_HL2 = 500
        self.N_NODES_HL3 = 500

        self.N_CLASSES = 10
        self.BATCH_SIZE = 100
        self.HM_EPOCHS = 10

        self.x = tf.placeholder('float', [None, 784])
        self.y = tf.placeholder('float')

        self.hidden_1_layer = {
            'f_fum': self.N_NODES_HL1,
            'weights': tf.Variable(tf.random_normal([784, self.N_NODES_HL1])),
            'biases': tf.Variable(tf.random_normal([self.N_NODES_HL1]))
        }

        self.hidden_2_layer = {
            'f_fum': self.N_NODES_HL2,
            'weights': tf.Variable(tf.random_normal([self.N_NODES_HL1, self.N_NODES_HL2])),
            'biases': tf.Variable(tf.random_normal([self.N_NODES_HL2]))
        }

        self.hidden_3_layer = {
            'f_fum': self.N_NODES_HL3,
            'weights': tf.Variable(tf.random_normal([self.N_NODES_HL2, self.N_NODES_HL3])),
            'biases': tf.Variable(tf.random_normal([self.N_NODES_HL3]))
        }

        self.output_layer = {
            'f_fum':None,
            'weights':tf.Variable(tf.random_normal([self.N_NODES_HL3, self.N_CLASSES])),
            'biases':tf.Variable(tf.random_normal([self.N_CLASSES]))
        }

        # saves the variables for checkpoints
        self.saver = tf.train.Saver();

        self.train_x, self.train_y = create_features_and_labels("""SELECT * FROM images LIMIT 100""")
        self.test_x, self.test_y = create_features_and_labels("""SELECT *  FROM images ORDER BY random() LIMIT 100""")

        self.accuracy = 0;

    def create_nnet_model(self, data):
        print("creating model...")
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['weights']), self.hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['weights']), self.hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, self.hidden_3_layer['weights']), self.hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3, self.output_layer['weights']) + self.output_layer['biases']
        print("done creating model")

        return output

    def train(self):
        train_x = self.train_x
        train_y = self.train_y
        test_x = self.test_x
        test_y = self.test_y

        prediction = self.create_nnet_model(self.x)
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y) )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(self.HM_EPOCHS):
                epoch_loss = 0
                i=0
                while i < len(train_x):
                    start = i
                    end = i + self.BATCH_SIZE
                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])

                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x,
                                                                  self.y: batch_y})
                    epoch_loss += c
                    i += self.BATCH_SIZE

                self.saver.save(sess, "./tmp/model.ckpt")
                print('Epoch', epoch + 1, 'completed out of', self.HM_EPOCHS, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            self.accuracy = accuracy.eval({self.x:test_x, self.y:test_y})

# train_x, train_y = create_features_and_labels("""SELECT * FROM images LIMIT 100""")
# test_x, test_y = create_features_and_labels("""SELECT *  FROM images ORDER BY random() LIMIT 100""")

a = NeuralNetModel()
a.train()
print("A's accuracy: ", a.accuracy)
