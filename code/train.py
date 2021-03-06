from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import numpy as np

from qa_model import QASystem, Config
from qa_encoder_decoder import BasicAffinityEncoder, BasicLSTMClassifyDecoder
from qa_utils import prepare_data
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_integer("subsample", None, "For testing purpose, subsample a portion of data to feedinto model")
tf.app.flags.DEFINE_integer("context_max_length", 200, "Trim or pad context paragraph to this length.")
tf.app.flags.DEFINE_integer("question_max_length", 30, "Trim or pad question to this length.")
tf.app.flags.DEFINE_integer("eval_freq", 5, "For how many training epochs do we evaluate the model once.")
tf.app.flags.DEFINE_boolean("save_parameters", True, "Whether to save model parameters or not.")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Learning rate decay rate.")
tf.app.flags.DEFINE_float("train_rate", 0.75, "The portion of training data seen in each epoch.")
tf.app.flags.DEFINE_integer("npairs", 3, "Number of encoder-decoder pairs to ensemble.")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def main(_):
    dataset = None
    embed_path = FLAGS.embed_path or pjoin("data", "squad")
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    ## collect input arguments to construct config object
    initial_config = {}
    initial_config['learning_rate'] = FLAGS.learning_rate
    initial_config['dropout'] = FLAGS.dropout
    initial_config['batch_size'] = FLAGS.batch_size
    initial_config['epochs'] = FLAGS.epochs
    initial_config['state_size'] = FLAGS.state_size
    initial_config['embedding_size'] = FLAGS.embedding_size
    initial_config['data_dir'] = FLAGS.data_dir
    initial_config['optimizer'] = FLAGS.optimizer
    initial_config["embed_path"] = FLAGS.embed_path
    initial_config['c_max_length'] = FLAGS.context_max_length
    initial_config['q_max_length'] = FLAGS.question_max_length
    initial_config['eval_freq'] = FLAGS.eval_freq
    initial_config['decay_rate'] = FLAGS.decay_rate
    initial_config['train_rate'] = FLAGS.train_rate
    initial_config['npairs'] = FLAGS.npairs

    config = Config(initial_config)

    ## load training and validation data
    c_max_length = FLAGS.context_max_length
    q_max_length = FLAGS.question_max_length

    train_contexts, train_questions, train_context_ids, train_context_masks, train_question_ids, train_start_ids, train_end_ids, train_answers = prepare_data(
        data_dir = FLAGS.data_dir,
        c_max_length = c_max_length,
        q_max_length = q_max_length,
        train_val = "train",
        sample_size = FLAGS.subsample)

    val_contexts, val_questions, val_context_ids, val_context_masks, val_question_ids, val_start_ids, val_end_ids, val_answers = prepare_data(
        data_dir = FLAGS.data_dir,
        c_max_length = c_max_length,
        q_max_length = q_max_length,
        train_val = "val",
        sample_size = FLAGS.subsample)

    ## === pack data to feed into model ===
    dataset = {}
    dataset['contexts'] = np.asarray(train_context_ids)
    dataset['questions'] = np.asarray(train_question_ids)
    dataset['start_labels'] = np.asarray(train_start_ids)
    dataset['end_labels'] = np.asarray(train_end_ids)
    dataset['original_contexts'] = train_contexts
    dataset['answers'] = train_answers

    dataset['val_contexts'] = np.asarray(val_context_ids)
    dataset['val_questions'] = np.asarray(val_question_ids)
    dataset['val_start_labels'] = np.asarray(val_start_ids)
    dataset['val_end_labels'] = np.asarray(val_end_ids)
    dataset['val_original_contexts'] = val_contexts
    dataset['val_answers'] = val_answers

    encoders, decoders = [], []
    for idx in xrange(FLAGS.npairs):
        encoders.append(BasicAffinityEncoder(config))
        decoders.append(BasicLSTMClassifyDecoder(config))

    qa = QASystem(encoders, decoders, config)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, save_train_dir, FLAGS.save_parameters)

if __name__ == "__main__":
    tf.app.run()
