# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:08:27 2018

@author: 74284
"""

import os
import time
import pickle
from collections import OrderedDict
from elmoformanylangs import Embedder
from use_elmo import get_elmo
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager
from convert_feature_tag import convert_feature_tag
from convert_evaluate_input import evaluate_inputs

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("elmo_dim",    1024,        "Embedding size for elmo")
flags.DEFINE_integer("layer1_dim",    1024,        "Embedding size for elmo")
flags.DEFINE_integer("layer2_dim",    1024,        "Embedding size for elmo")
flags.DEFINE_integer("layer3_dim",    1024,        "Embedding size for elmo")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    32,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     os.path.join("data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
#flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["elmo_dim"] = FLAGS.elmo_dim
    config["layer1_dim"] = FLAGS.layer1_dim
    config["layer2_dim"] = FLAGS.layer2_dim
    config["layer3_dim"] = FLAGS.layer3_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config

def test_evaluate(text):
    '''
    :param text:list of string
    :return:NER_predict_result of text
    '''
    start = time.clock()
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
        
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, config, id_to_char, logger, False)
        feature_load,str_load=get_elmo(text)
        input_batch=evaluate_inputs(str_load,char_to_id,feature_load)
        result = model.evaluate_line(sess, input_batch, id_to_tag)
    end = time.clock()
    #print(result)
    print ("read: %f s" % (end - start))
    tf.reset_default_graph()
    return result

    
