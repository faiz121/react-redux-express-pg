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
        self.train_x, self.train_y = create_features_and_labels("""SELECT * FROM images LIMIT 5000""")
        self.test_x, self.test_y = create_features_and_labels("""SELECT *  FROM images ORDER BY random() LIMIT 1000""")

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

            print('prediction: ', prediction)
            print('tf.argmax(prediction, 1): ', tf.argmax(prediction, 1))

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            self.accuracy = accuracy.eval({self.x:test_x, self.y:test_y})

    def standardize_input_data(self, input):
        if any(item > 1 for item in input):
            print("standardizing input...")
            return np.asfarray(np.array(input)) / 255.0 * 1
        else:
            print("input already standardized...")
            return np.array(input)

    # np.shape(input_data) === (1, 784)
    def run_model(self, input_data):
        prediction = self.create_nnet_model(self.x)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.saver.restore(sess, "./tmp/model.ckpt")
            print("input_data: ", input_data)
            features = self.standardize_input_data(input_data)
            print("features: ", features)

            placeholder_y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            one_hot_result = prediction.eval(feed_dict={self.x: [features], self.y: [placeholder_y]})
            result = sess.run(tf.argmax(one_hot_result, 1))
            print("one_hot_result: ", one_hot_result)
            print("result: ", result)


model = NeuralNetModel()
model.train()
print("model accuracy: ", model.accuracy)
# should output 7
model.run_model([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027450980392156862, 0.47058823529411764, 0.5725490196078431, 0.5725490196078431, 0.5725490196078431, 0.5725490196078431, 0.6627450980392157, 0.8156862745098039, 0.996078431372549, 1.0, 0.8156862745098039, 0.996078431372549, 0.5607843137254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35294117647058826, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9254901960784314, 0.12156862745098039, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.023529411764705882, 0.7568627450980392, 0.9725490196078431, 0.9725490196078431, 0.9725490196078431, 0.9725490196078431, 0.9725490196078431, 0.6352941176470588, 0.5450980392156862, 0.5450980392156862, 0.7176470588235294, 0.9921568627450981, 0.7686274509803922, 0.023529411764705882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3843137254901961, 0.9921568627450981, 0.7372549019607844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6980392156862745, 0.9921568627450981, 0.4235294117647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5882352941176471, 0.9764705882352941, 0.8784313725490196, 0.03529411764705882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.054901960784313725, 0.8666666666666667, 0.9921568627450981, 0.4666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23137254901960785, 0.9921568627450981, 0.5529411764705883, 0.10980392156862745, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7176470588235294, 0.9921568627450981, 0.18823529411764706, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23529411764705882, 0.9725490196078431, 0.9921568627450981, 0.18823529411764706, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5137254901960784, 0.9921568627450981, 0.6549019607843137, 0.0392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10980392156862745, 0.8941176470588236, 0.9882352941176471, 0.30980392156862746, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4980392156862745, 0.9921568627450981, 0.9098039215686274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6352941176470588, 0.9921568627450981, 0.5647058823529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2901960784313726, 0.9803921568627451, 0.8, 0.047058823529411764, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7764705882352941, 0.9921568627450981, 0.6313725490196078, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9137254901960784, 0.9921568627450981, 0.2823529411764706, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9137254901960784, 0.7294117647058823, 0.00392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9137254901960784, 0.3607843137254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14901960784313725, 0.9215686274509803, 0.20784313725490197, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
