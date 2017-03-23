from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from qa_utils import load_glove
from os.path import join as pjoin

logging.basicConfig(level=logging.INFO)


class Config(object):
    def __init__(self, initial_config):
        self.learning_rate = initial_config['learning_rate']
        self.dropout = initial_config['dropout']
        self.batch_size = initial_config['batch_size']
        self.epochs = initial_config['epochs']
        self.state_size = initial_config['state_size']
        self.embedding_size = initial_config['embedding_size']
        self.data_dir = initial_config['data_dir']
        self.optimizer = initial_config['optimizer']
        self.embed_path = initial_config["embed_path"]
        self.c_max_length = initial_config['c_max_length']
        self.q_max_length = initial_config['q_max_length']
        self.eval_freq = initial_config['eval_freq']
        self.decay_rate = initial_config['decay_rate']
        self.train_rate = initial_config['train_rate']
        self.npairs = initial_config['npairs']


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class QASystem(object):
    def __init__(self, encoders, decoders, config, *args):
        """
        Initializes your System

        :param encoders: a list of encoders constructed in train.py
        :param decoders: a list of decoders constructed in train.py
        :param config: Config object containing all configurations to the model 
        """
        # ==== set up basic stuff ====
        self.config = config
        self.glove = load_glove(
            self.config.data_dir,
            self.config.embedding_size)

        self.encoders = encoders
        self.decoders = decoders

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32,
            shape = [None, self.config.c_max_length],
            name = "context")
        self.question_placeholder = tf.placeholder(tf.int32,
            shape = [None, self.config.q_max_length],
            name = "question")
        self.start_label_placeholder = tf.placeholder(tf.int32,
            shape = [None],
            name = "start_label")
        self.end_label_placeholder = tf.placeholder(tf.int32,
            shape = [None],
            name = "end_label")
        self.dropout_placeholder = tf.placeholder(tf.float32,
            shape = [],
            name = "dropout")

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_optimization()

        self.saver = tf.train.Saver()


    def create_feed_dict(self, input_contexts, input_questions,
        input_start_labels = None, input_end_labels = None, input_dropout = None):
        feed_dict = {}
        feed_dict[self.context_placeholder] = input_contexts
        feed_dict[self.question_placeholder] = input_questions

        if input_start_labels is not None:
            feed_dict[self.start_label_placeholder] = input_start_labels
        
        if input_end_labels is not None:
            feed_dict[self.end_label_placeholder] = input_end_labels

        if input_dropout is not None:
            feed_dict[self.dropout_placeholder] = input_dropout
        else:
            feed_dict[self.dropout_placeholder] = 0.0
        return feed_dict


    def setup_system(self):
        """
        Connects the encoders and decoders in the system. Each encoder-decoder pair receives the
        same input.
        """
        dataset_encoder = {}
        dataset_encoder['contexts'] = self.context_embeddings
        dataset_encoder['questions'] = self.question_embeddings
        dataset_encoder['dropout'] = self.dropout_placeholder

        self.start_preds, self.end_preds = [], []
        for idx in xrange(self.config.npairs):
            with vs.variable_scope("setpup{}".format(idx)):
                start_pred, end_pred = self.decoders[idx].decode(self.encoders[idx].encode(dataset_encoder))
                self.start_preds.append(start_pred)
                self.end_preds.append(end_pred)

    def setup_loss(self):
        """
        Set loss computation. Losses from different encoder-decoder pairs are collected in a list
        """
        self.losses = []
        for idx in xrange(self.config.npairs):
            with vs.variable_scope("setupLoss{}".format(idx)):
                with vs.variable_scope("loss_start"):
                    loss_s = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(self.start_preds[idx], self.start_label_placeholder))
                with vs.variable_scope("loss_end"):
                    loss_e = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(self.end_preds[idx], self.end_label_placeholder))
                self.losses.append(loss_s + loss_e)

    def setup_optimization(self):
        """
        Set optimization operations. Each encoder-decoder pair is individually optimized.
        """
        with vs.variable_scope("optimization"):

            self.learning_rate = self.config.learning_rate
            self.optimizer = get_optimizer(self.config.optimizer)(self.learning_rate)

            self.opts = []
            for idx in xrange(self.config.npairs):
                self.opts.append(self.optimizer.minimize(self.losses[idx]))

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        """
        with vs.variable_scope("embeddings"):
            ## we don't train word embeddings
            self.glove_dictionary = tf.constant(self.glove, dtype = tf.float32)
            self.context_embeddings = tf.nn.embedding_lookup(
                self.glove_dictionary, self.context_placeholder)
            self.question_embeddings = tf.nn.embedding_lookup(
                self.glove_dictionary, self.question_placeholder)

    def optimize(self, session, dataset):
        """
        Takes in actual data and perform an optimizatio step. Each encoder-decoder pair
        is trained in one optimize step.
        """
        contexts = dataset['contexts']
        questions = dataset['questions']
        start_labels = dataset['start_labels']
        end_labels = dataset['end_labels']

        input_feed = self.create_feed_dict(
            input_contexts = contexts,
            input_questions = questions,
            input_start_labels = start_labels,
            input_end_labels = end_labels,
            input_dropout = self.config.dropout)

        output_feed = self.losses + self.opts
        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, dataset):
        """
        Test model performance my computing loss with gien input
        """
        contexts = dataset['contexts']
        questions = dataset['questions']
        start_labels = dataset['start_labels']
        end_labels = dataset['end_labels']
        input_feed = self.create_feed_dict(
            input_contexts = contexts,
            input_questions = questions,
            input_start_labels = start_labels,
            input_end_labels = end_labels)

        output_feed = self.losses

        outputs = session.run(output_feed, input_feed)
        outputs = np.mean(outputs)

        return outputs

    def decode(self, session, dataset):
        """
        Produce predictions for answer starting point and ending poitn's
        probability distributions given actual input.
        """
        contexts = dataset['contexts']
        questions = dataset['questions']
        input_feed = self.create_feed_dict(
            input_contexts = contexts,
            input_questions = questions)

        output_feed = self.start_preds + self.end_preds

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, dataset):
        """
        Given actual input, produce predictions for answer's starting and ending positions.
        :return:
        a_s: starting positions as an numpy array, with shape = [batch_size]
        a_e: ending positions as an numpy array, with shape = [batch_size]
        """
        predictions = self.decode(session, dataset)
        start_preds = predictions[: self.config.npairs]
        end_preds = predictions[self.config.npairs : ]

        start_preds = np.sum(start_preds, axis = 0)
        end_preds = np.sum(end_preds, axis = 0)

        a_s = np.argmax(start_preds, axis=1)
        a_e = np.argmax(end_preds, axis=1)

        return (a_s, a_e)

    def validate(self, sess, val_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation loss is.

        This method calls self.test() which explicitly calculates validation cost.

        :return:
        valid_cost: the calculated loss for the given validation set.
        """
        context_ids = val_dataset['contexts']
        question_ids = val_dataset['questions']
        start_labels = val_dataset['start_labels']
        end_labels = val_dataset['end_labels']

        ## create batches to calculate validation loss to prevent OOM.
        batch_size = self.config.batch_size
        input_size = len(start_labels)
        n_batches = int(input_size / batch_size) + 1
        data_batches = []
        for batch_id in xrange(n_batches):
            id_h = batch_id * batch_size
            id_t = (batch_id + 1) * batch_size
            d = {}
            d['contexts'] = context_ids[id_h : id_t]
            d['questions'] = question_ids[id_h : id_t]
            d['start_labels'] = start_labels[id_h : id_t]
            d['end_labels'] = end_labels[id_h : id_t]
            data_batches.append(d)

        valid_cost = 0.

        for data_batch in data_batches:
            n_samples_in_batch = len(data_batch['start_labels'])
            cost = self.test(sess, data_batch)
            valid_cost += cost * n_samples_in_batch
        valid_cost /= input_size

        return valid_cost

    def evaluate_answer(self, session, dataset, sample = 100, log = False):
        """
        Evaluate the model's performance the mean of F1 and Exact Match (EM)
        with the set of true answer labels

        Only a sample of input data is taken for the purpose of evaluation.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        f1: average F1 score over "sample" number of samples over the input dataset
        f1: average EM score over "sample" number of samples over the input dataset
        """
        f1 = 0.
        em = 0.

        ## === take samples ===
        context_ids = dataset['contexts']
        question_ids = dataset['questions']
        original_contexts = dataset['original_contexts']
        answers = dataset['answers']

        assert len(context_ids) == len(question_ids) == len(original_contexts) == len(answers)

        input_size = len(answers)
        s_context_ids, s_question_ids, s_contexts, s_answers = [], [], [], []
        if input_size > sample:
            sample_ids = sorted(np.random.choice(range(input_size), size = sample, replace = False))
            for idx in sample_ids:
                s_context_ids.append(context_ids[idx])
                s_question_ids.append(question_ids[idx])
                s_contexts.append(original_contexts[idx])
                s_answers.append(answers[idx])

            original_contexts = s_contexts
            answers = s_answers
        else:
            s_context_ids = context_ids
            s_question_ids = question_ids

        sampled_dataset = {}
        sampled_dataset['contexts'] = s_context_ids
        sampled_dataset['questions'] = s_question_ids

        ## === get starting and ending positions ===
        tic = time.time()
        a_s, a_e = self.answer(session, sampled_dataset)
        for start, end, context, ans in zip(a_s, a_e, original_contexts, answers):
            pred_ans = context[start: end + 1]
            pred_ans = " ".join(pred_ans)
            true_ans = " ".join(ans)
            f1 += f1_score(pred_ans, true_ans)
            em += exact_match_score(pred_ans, true_ans)

        f1 = f1 / len(answers)
        em = em / len(answers)
        toc = time.time()

        if log:
            logging.info("==== Evaluation took {} (sec) with F1: {}, EM: {}, for {} samples".format(toc - tic, f1, em, min(input_size, sample)))

        return f1, em

    def train(self, session, dataset, train_dir, save_parameters = True):
        """
        Main training loop

        Training is done in batches. Note that batch size is crutial in training performance.
        A large batch size boosts training efficiency and also allows optimizers to perform their full capacity.
        Choose a batch size that balances training efficiency and hardware capability.

        :param session: it should be passed in from train.py
        :param dataset: a representation of input data
        :param train_dir: path to the directory where you should save the model checkpoint
        """

        ## ===== Both training and validation sets are passed into train function ====
        ## process dataset
        val_dataset = {}
        val_dataset['contexts'] = dataset['val_contexts']
        val_dataset['questions'] = dataset['val_questions']
        val_dataset['start_labels'] = dataset['val_start_labels']
        val_dataset['end_labels'] = dataset['val_end_labels']
        val_dataset['original_contexts'] = dataset['val_original_contexts']
        val_dataset['answers'] = dataset['val_answers']

        ## create training batch with one training dictionary per batch
        context_ids = dataset['contexts']
        question_ids = dataset['questions']
        start_labels = dataset['start_labels']
        end_labels = dataset['end_labels']

        batch_size = self.config.batch_size
        input_size = len(start_labels)
        n_batches = int(input_size / batch_size) + 1
        data_batches = []
        for batch_id in xrange(n_batches):
            id_h = batch_id * batch_size
            id_t = (batch_id + 1) * batch_size
            d = {}
            d['contexts'] = context_ids[id_h : id_t]
            d['questions'] = question_ids[id_h : id_t]
            d['start_labels'] = start_labels[id_h : id_t]
            d['end_labels'] = end_labels[id_h : id_t]
            data_batches.append(d)
        logging.info("Training with {} batches with batch size: {}".format(n_batches, batch_size))

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        ## ==== training process ====
        for epoch in xrange(self.config.epochs):
            tic = time.time()
            loss = 0.
            ## randomize batch training order
            ## only a portion of training samples were feeded in each epoch
            batch_order = sorted(np.random.choice(range(n_batches), size = int(n_batches * self.config.train_rate), replace = False))
            trained_size = 0
            for batch_id in batch_order:
                data_batch = data_batches[batch_id]
                n_samples_in_batch = len(data_batch['start_labels'])
                output = self.optimize(session, data_batch)
                batch_loss = np.mean(output[:self.config.npairs])
                loss += batch_loss * n_samples_in_batch
                trained_size += n_samples_in_batch
            loss /= trained_size
            
            ## evaluate validation loss
            valid_loss = self.validate(session, val_dataset)
            toc = time.time()
            logging.info("====================== Epoch {} took {} (sec) with training loss: {}, validation loss: {}, exit learning_rate: {}".format(epoch + 1, toc - tic, loss, valid_loss, self.config.learning_rate))

            ## learning rate exponential decay
            self.config.learning_rate *= self.config.decay_rate

            ## evaluate model performance
            eval_freq = self.config.eval_freq
            if epoch % eval_freq == (eval_freq - 1):
                logging.info("==== Evaluating training set ====")
                self.evaluate_answer(session, dataset, log = True)
                logging.info("==== Evaluating validation set ====")
                self.evaluate_answer(session, val_dataset, log = True)

                ## save model parameters
                if save_parameters:
                    save_path = self.saver.save(session, pjoin(train_dir, "model" + str(int(time.time())) + ".ckpt"))
                    logging.info("** Saved parameters to {}".format(save_path))



