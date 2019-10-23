# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from convert_feature_tag import convert_feature_tag

import rnncell as rnn
from utils import result_to_json
from data_utils import create_input, iobes_iob


class Model(object):
    def __init__(self, config, is_train=True):

        self.config = config
        self.is_train = is_train
        
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]
        self.layer1_dim=config["layer1_dim"]
        self.elmo_dim=config["elmo_dim"]
        self.layer2_dim=config["layer2_dim"]
        self.layer3_dim=config["layer3_dim"]
        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        
        

        # add placeholders for the model

        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")
        self.layer1_inputs=tf.placeholder(dtype=tf.float32,
                                         shape=[None,None,1024],
                                         name="layer1Inputs")
        self.layer2_inputs=tf.placeholder(dtype=tf.float32,
                                         shape=[None,None,1024],
                                         name="layer2Inputs")
        self.layer3_inputs=tf.placeholder(dtype=tf.float32,
                                         shape=[None,None,1024],
                                         name="layer3Inputs")
        self.elmo_inputs=tf.placeholder(dtype=tf.float32,
                                         shape=[None,None,1024],
                                         name="elmoInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        
        
        #Add model type by crownpku， bilstm or idcnn
        self.model_type = config['model_type']
        #parameters for idcnn
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim 
        self.embedding_dim =self.elmo_dim
        self.repeat_times = 4
        self.cnn_output_width = 0
        
        

        # embeddings for chinese character and segmentation representation
        embedding = self.attention_layer(self.layer1_inputs,self.layer2_inputs,self.layer3_inputs)
        #embedding=GroupNorm(embedding)

        if self.model_type == 'bilstm':
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags
            self.logits = self.project_layer_bilstm(model_outputs)
        
        elif self.model_type == 'idcnn':
            # apply dropout before feed to idcnn layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # ldcnn layer
            model_outputs = self.IDCNN_layer(model_inputs)

            # logits for tags
            self.logits = self.project_layer_idcnn(model_outputs)
        
        else:
            raise KeyError

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        

        
    def attention_layer(self,layer1_inputs,layer2_inputs,layer3_inputs):
        layer1=self.layer1_inputs
        layer2=self.layer2_inputs
        layer3=self.layer3_inputs
        #layer1=GroupNorm(layer1)
        #layer2=GroupNorm(layer2)
        #layer3=GroupNorm(layer3)
        hidden_size =layer1.shape[2].value
        attention_size=layer1.shape[2].value
        w1_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        w2_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        w3_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
#        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#        u_omega = tf.Variable(tf.random_normal([attention_size,hidden_size], stddev=0.1))
        #w1_omega=tf.cast(w1_omega, tf.float64) 
        #w2_omega=tf.cast(w2_omega, tf.float64) 
        #w3_omega=tf.cast(w3_omega, tf.float64) 
#        b_omega=tf.cast(b_omega,tf.float64)
#        u_omega=tf.cast(u_omega,tf.float64)
        
        v1=tf.tanh(tf.tensordot(layer1, w1_omega, axes=1))
        v2=tf.tanh(tf.tensordot(layer2, w2_omega, axes=1))
        v3=tf.tanh(tf.tensordot(layer3, w3_omega, axes=1))
        alphas1=tf.exp(v1)/ (tf.exp(v1)+tf.exp(v2)+tf.exp(v3))
        alphas2=tf.exp(v2)/ (tf.exp(v1)+tf.exp(v2)+tf.exp(v3))
        alphas3=tf.exp(v3)/ (tf.exp(v1)+tf.exp(v2)+tf.exp(v3))
        elmo_output= tf.multiply(layer1, alphas1)+tf.multiply(layer2, alphas2)+tf.multiply(layer3, alphas3)
        
        return elmo_output
    
 

    def embedding_layer(self,elmo_inputs,config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """

        embedding = []
        with tf.variable_scope("elmo_embedding"), tf.device('/cpu:0'):
            embedding.append(self.elmo_inputs)
            embed =embedding 
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)
    
    #IDCNN layer 
    def IDCNN_layer(self, model_inputs, 
                    name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, cnn_output_width]
        """
        model_inputs = tf.expand_dims(model_inputs, 1)
        #print(model_inputs.shape)
        reuse = False
        if not self.is_train:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            
            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=tf.AUTO_REUSE):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        #print(conv.shape)
                        #conv=tf.contrib.layers.group_norm(conv)
                        #conv=GroupNorm(conv)
                        conv=tf.contrib.layers.layer_norm(conv)
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        #print(conv.shape)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
                        #print(layerInput.shape)
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            #print(finalOut.shape)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            #print(totalWidthForLastDim)
            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            #print(finalOut.shape)
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])
    
    #Project layer for idcnn by crownpku
    #Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        strs,chars,layer1,layer2,layer3, tags = batch
        
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.layer1_inputs: np.asarray(layer1),
            self.layer2_inputs: np.asarray(layer2),
            self.layer3_inputs: np.asarray(layer3),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(True, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            feed_dict[self.dropout]=1.0
            lengths, logits,testloss = sess.run([self.lengths, self.logits,self.loss], feed_dict)
            return lengths, logits,testloss

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_num,char_to_id,tag_to_id, id_to_tag,data_type):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        total_loss=[]
        for j in range(data_num):
            if data_type=="test":
                tag_path='NER_test/test_label'+str(j+1)+'.npy'
                feature_path='NER_test/elmo_result'+str(j+1)+'.npy'
                str_path='NER_test/test_sentence'+str(j+1)+'.npy'
            if data_type=="dev":
                tag_path='NER_dev/dev_label'+str(j+1)+'.npy'
                feature_path='NER_dev/elmo_result'+str(j+1)+'.npy'
                str_path='NER_dev/dev_sentence'+str(j+1)+'.npy'
            tag_load=np.load(tag_path)
            feature_load=np.load(feature_path)
            str_load=np.load(str_path)
            data_tags=[]
            for sen in tag_load:
                tag=[]
                for item in sen:
                    tag.append(tag_to_id[item])
                data_tags.append(tag)
            batch=convert_feature_tag(str_load,char_to_id,feature_load,data_tags)
            strings = batch[0]
            tags = batch[5]
            lengths, scores,testloss = self.run_step(sess, False, batch)
            total_loss.append(testloss)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results,total_loss
    
    def evaluate_run(self, sess,is_train, batch):
        """
        :param sess: session to run the evaluate_batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:       
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.evaluate_run(sess, False,inputs)
        strings=inputs[0]
        batch_paths = self.decode(scores, lengths, trans)
        result=[]
        for i in range(len(strings)):
            char_input=strings[i][:lengths[i]]
            #print(char_input)
            tags = [id_to_tag[idx] for idx in batch_paths[i][:lengths[i]]]
            json_result=result_to_json(char_input, tags)
            print(json_result)
            result.append(json_result)
        return result
