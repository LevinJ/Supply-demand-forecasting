import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
import numpy as np
import logging
from bokeh.util.logconfig import level
import sys
from utility.tfbasemodel import TFModel
from preprocess.preparedata import PrepareData
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from evaluation.sklearnmape import mean_absolute_percentage_error

class DididNeuralNetowrk(TFModel, PrepareData):
    def __init__(self):
        TFModel.__init__(self)
        PrepareData.__init__(self)
        self.num_steps = 500
        self.batch_size = 128
        self.summaries_dir = '/tmp/didi'
        return
    def add_visualize_node(self):
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.summaries_dir+ '/train',
                                        self.graph)
        self.test_writer = tf.train.SummaryWriter(self.summaries_dir + '/test')

        return
    def get_input(self):
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        self.x_train, self.y_train,self.x_validation,self.y_validation = self.get_train_validationset(-2)
        self.x_train, self.y_train,self.x_validation,self.y_validation = self.x_train.as_matrix(), self.y_train.as_matrix().reshape((-1,1)),\
                                                                         self.x_validation.as_matrix(),self.y_validation.as_matrix().reshape((-1,1))
#         self.x_train, self.y_train,self.x_validation,self.y_validation = self.x_train.astype(np.float32), self.y_train.astype(np.float32),\
#                                                                          self.x_validation.astype(np.float32),self.y_validation.astype(np.float32)
        sc = MinMaxScaler()
        sc.fit(self.x_train)
        self.x_train= sc.transform(self.x_train)
        self.x_validation= sc.transform(self.x_validation)
        
        self.inputlayer_num = len(self.usedFeatures)
        self.outputlayer_num = 1
        
        # Input placehoolders
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, self.inputlayer_num], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, self.outputlayer_num ], name='y-input')
        self.keep_prob = tf.placeholder(tf.float32, name='drop_out')
        
        return
    def add_inference_node(self):
        #output node self.pred
        hidden1 = self.nn_layer(self.x, 500, 'layer1')
        dropped = self.dropout_layer(hidden1)
        
        hidden1 = self.nn_layer(dropped, 300, 'layer2')
        dropped = self.dropout_layer(hidden1)
        
        self.y_pred = self.nn_layer(dropped, self.outputlayer_num , 'layer3')
        return
    def add_loss_node(self):
        #output node self.loss
 
        self.__add_mape_loss()
        return
    def __add_mse_loss(self):
        with tf.name_scope('loss'):
            diff = tf.square(self.y_true - self.y_pred)
            with tf.name_scope('mse'):
                self.loss = tf.reduce_mean(diff)
            tf.scalar_summary('mse', self.loss)
        return
    def __add_mape_loss(self):
        with tf.name_scope('loss'):
            diff = tf.abs((self.y_true - self.y_pred)/self.y_true)
            with tf.name_scope('mape'):
                self.loss = tf.reduce_mean(tf.cast(diff, tf.float32))
            tf.scalar_summary('loss', self.loss)
        return
    def add_optimizer_node(self):
        #output node self.train_step
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        return
    def add_accuracy_node(self):
        #output node self.accuracy
        with tf.name_scope('evaluationmetrics'):
            with tf.name_scope('error_square'):
                error_square = tf.abs((self.y_true - self.y_pred)/self.y_true)
            with tf.name_scope('mape'):
                self.accuracy = tf.reduce_mean(tf.cast(error_square, tf.float32))
            tf.scalar_summary('mape', self.accuracy)
        return
    def add_evalmetrics_node(self):
        self.add_accuracy_node()
        return
    def feed_dict(self,feed_type):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if feed_type == "train":
            xs, ys = self.get_next_batch(self.x_train, self.y_train, self.batch_size)
            k = self.dropout
            return {self.x: xs, self.y_true: ys, self.keep_prob: k}
        if feed_type == "validation":
            xs, ys = self.x_validation, self.y_validation
            k = 1.0
            return {self.x: xs, self.y_true: ys, self.keep_prob: k}
        if feed_type == "validation_wholetrain":
            xs, ys = self.x_train, self.y_train
            k = 1.0
            return {self.x: xs, self.y_true: ys, self.keep_prob: k}
        # Now we are feeding test data into the neural network
        if feed_type == "test":
            xs= self.x_test
            k = 1.0
            return {self.x: xs, self.keep_prob: k}

    def run_graph(self):
        logging.debug("computeGraph")
        with tf.Session(graph=self.graph) as sess:
            tf.initialize_all_variables().run()
            logging.debug("Initialized")
            for step in range(self.num_steps + 1):
                summary, _ , train_loss, train_metrics= sess.run([self.merged, self.train_step, self.loss, self.accuracy], feed_dict=self.feed_dict("train"))
                self.train_writer.add_summary(summary, step)
                
                if step % 10 == 0:
                    summary, validation_loss, validation_metrics = sess.run([self.merged, self.loss, self.accuracy], feed_dict=self.feed_dict("validation"))
                    self.test_writer.add_summary(summary, step)
#                     loss_train = sess.run(self.loss, feed_dict=self.feed_dict("validation_wholetrain"))
                    logging.info("Step {}/{}, train/test: {:.3f}/{:.3f}, train/test loss: {:.3f}/{:.3f}".format(step, self.num_steps, train_metrics, validation_metrics,\
                                                                                                                train_loss, validation_loss))
    
#                     y_pred = sess.run(self.y_pred, feed_dict=self.feed_dict("validation"))
#                     logging.info("validation mape :{:.3f}".format(mean_absolute_percentage_error(self.y_validation.reshape(-1), y_pred.reshape(-1))))
        return


if __name__ == "__main__":   
    obj= DididNeuralNetowrk()
    obj.run()