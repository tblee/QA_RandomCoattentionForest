from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Config, QASystem
from qa_encoder_decoder import BasicAffinityEncoder, BasicLSTMClassifyDecoder
from qa_utils import pad_and_trim_sentence_with_mask
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

tf.app.flags.DEFINE_integer("subsample", None, "For testing purpose, subsample a portion of data to feedinto model")
tf.app.flags.DEFINE_integer("context_max_length", 200, "Trim or pad context paragraph to this length.")
tf.app.flags.DEFINE_integer("question_max_length", 30, "Trim or pad question to this length.")
tf.app.flags.DEFINE_integer("eval_freq", 5, "For how many training epochs do we evaluate the model once.")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Learning rate decay rate.")
tf.app.flags.DEFINE_float("train_rate", 0.75, "The portion of training data seen in each epoch.")
tf.app.flags.DEFINE_integer("npairs", 3, "Number of encoder-decoder pairs to ensemble.")


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


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


def generate_answers(sess, model, input_dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    ## === process input data ===
    context_data = input_dataset['contexts']
    question_data = input_dataset['questions']
    uuids = input_dataset['uuids']

    context_data = map(lambda s: map( int, s.strip('\n').split(' ')), context_data)
    question_data = map(lambda s: map( int, s.strip('\n').split(' ')), question_data)

    context_data = map(lambda s: pad_and_trim_sentence_with_mask(s, FLAGS.context_max_length)[0], context_data)
    question_data = map(lambda s: pad_and_trim_sentence_with_mask(s, FLAGS.question_max_length)[0], question_data)

    ## === extract model prediction in batches to prevent system overload ===
    input_size = len(uuids)
    batch_size = FLAGS.batch_size
    n_batches = int(input_size / batch_size) + 1
    a_s, a_e = np.asarray([], dtype = np.int32), np.asarray([], dtype = np.int32)
    for batch_id in xrange(n_batches):
        logging.info("Processing batch {}/{}...".format(batch_id + 1, n_batches))
        id_h = batch_id * batch_size
        id_t = (batch_id + 1) * batch_size
        data_batch = {}
        data_batch['contexts'] = np.asarray( context_data[id_h : id_t] )
        data_batch['questions'] = np.asarray( question_data[id_h : id_t] )
        batch_a_s, batch_a_e = model.answer(sess, data_batch)
        a_s = np.concatenate((a_s, batch_a_s))
        a_e = np.concatenate((a_e, batch_a_e))

    """
    dataset = {}
    dataset['contexts'] = np.asarray(context_data)
    dataset['questions'] = np.asarray(question_data)
    """

    ## === obtain predicted answers from model ===
    rev_dict = {}
    for idx, word in enumerate(rev_vocab):
        rev_dict[idx] = word

    #a_s, a_e = model.answer(sess, dataset)

    answers = {}
    for start, end, context, uuid in zip(a_s, a_e, context_data, uuids):
        ans = context[start : end + 1]
        ## eliminate padded words
        cur = len(ans) - 1
        while cur >= 0 and ans[cur] == qa_data.PAD_ID:
            cur -= 1
        ans = ans[:cur + 1]
        ans = map(lambda idx: rev_dict.get(idx, qa_data._UNK), ans)
        ans = " ".join(ans)
        answers[uuid] = ans

    return answers


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

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
    
    dataset = {}
    dataset['contexts'] = context_data
    dataset['questions'] = question_data
    dataset['uuids'] = question_uuid_data
    #dataset = (context_data, question_data, question_uuid_data)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    ## collect input arguments to construct config object
    initial_config = {}
    initial_config['learning_rate'] = FLAGS.learning_rate
    initial_config['max_gradient_norm'] = FLAGS.max_gradient_norm
    initial_config['dropout'] = FLAGS.dropout
    initial_config['batch_size'] = FLAGS.batch_size
    initial_config['epochs'] = FLAGS.epochs
    initial_config['state_size'] = FLAGS.state_size
    initial_config['output_size'] = FLAGS.output_size
    initial_config['embedding_size'] = FLAGS.embedding_size
    initial_config['data_dir'] = FLAGS.data_dir

    initial_config['optimizer'] = FLAGS.optimizer
    initial_config['print_every'] = FLAGS.print_every
    initial_config["keep"] = FLAGS.keep
    initial_config["embed_path"] = FLAGS.embed_path

    initial_config['c_max_length'] = FLAGS.context_max_length
    initial_config['q_max_length'] = FLAGS.question_max_length
    initial_config['eval_freq'] = FLAGS.eval_freq
    initial_config['decay_rate'] = FLAGS.decay_rate
    initial_config['train_rate'] = FLAGS.train_rate
    initial_config['npairs'] = FLAGS.npairs


    config = Config(initial_config)

    encoders, decoders = [], []
    for idx in xrange(FLAGS.npairs):
        encoders.append(BasicAffinityEncoder(config))
        decoders.append(BasicLSTMClassifyDecoder(config))

    #encoder = BasicAffinityEncoder(config)
    #decoder = BasicLSTMClassifyDecoder(config)

    qa = QASystem(encoders, decoders, config)

    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        initialize_model(sess, qa, train_dir)
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
