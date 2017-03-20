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
        self.max_gradient_norm = initial_config['max_gradient_norm']
        self.dropout = initial_config['dropout']
        self.batch_size = initial_config['batch_size']
        self.epochs = initial_config['epochs']
        self.state_size = initial_config['state_size']
        self.output_size = initial_config['output_size']
        self.embedding_size = initial_config['embedding_size']
        self.data_dir = initial_config['data_dir']
        self.optimizer = initial_config['optimizer']
        self.print_every = initial_config['print_every']
        self.keep = initial_config["keep"]
        self.embed_path = initial_config["embed_path"]
        self.c_max_length = initial_config['c_max_length']
        self.q_max_length = initial_config['q_max_length']
        self.eval_freq = initial_config['eval_freq']
        self.decay_rate = initial_config['decay_rate']


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class QASystem(object):
    def __init__(self, encoder, decoder, config, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # ==== set up basic stuff ====
        self.config = config
        self.glove = load_glove(
            self.config.data_dir,
            self.config.embedding_size)

        self.encoder = encoder
        self.decoder = decoder

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

        # ==== set up training/updating procedure ====
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
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        #raise NotImplementedError("Connect all parts of your system here!")
        dataset_encoder = {}
        dataset_encoder['contexts'] = self.context_embeddings
        dataset_encoder['questions'] = self.question_embeddings
        dataset_encoder['dropout'] = self.dropout_placeholder
        self.encoded = self.encoder.encode(dataset_encoder)
        self.start_preds, self.end_preds = self.decoder.decode(self.encoded)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss_start"):
            loss_s = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(self.start_preds, self.start_label_placeholder))
        with vs.variable_scope("loss_end"):
            loss_e = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(self.end_preds, self.end_label_placeholder))
        self.loss = loss_s + loss_e

    def setup_optimization(self):
        with vs.variable_scope("optimization"):
            """
            self.global_step = tf.Variable(0, trainable = False)
            self.learning_rate = tf.train.exponential_decay(
                self.config.learning_rate, self.global_step, self.config.decay_steps, 0.9)
            """

            self.learning_rate = self.config.learning_rate
            self.optimizer = get_optimizer(self.config.optimizer)(self.learning_rate)
            self.opt = self.optimizer.minimize(self.loss)
            #self.opt = self.optimizer.minimize(self.loss, global_step = self.global_step)
            
            """
            self.gradients = optimizer.compute_gradients(self.loss)
            self.output_gradients = []
            for grad, name in self.gradients:
                if grad is None:
                    self.output_gradients.append((tf.constant(0.0), name))
                else:
                    self.output_gradients.append((grad, name))
            self.opt = optimizer.apply_gradients(self.gradients)
            """

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
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
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
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

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.opt, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, dataset):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
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

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = self.loss

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, dataset):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        contexts = dataset['contexts']
        questions = dataset['questions']
        input_feed = self.create_feed_dict(
            input_contexts = contexts,
            input_questions = questions)

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.start_preds, self.end_preds]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, dataset):
        """
        :return:
        a_s: starting positions, with shape = [batch_size]
        a_e: ending positions, with shape = [batch_size]
        """
        start_preds, end_preds = self.decode(session, dataset)

        a_s = np.argmax(start_preds, axis=1)
        a_e = np.argmax(end_preds, axis=1)

        return (a_s, a_e)

    def validate(self, sess, val_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """

        context_ids = val_dataset['contexts']
        question_ids = val_dataset['questions']
        start_labels = val_dataset['start_labels']
        end_labels = val_dataset['end_labels']

        ## create batches to calculate validation loss to prevent flooded the model
        ## since we are not training parameters, take a larger batch size
        batch_size = self.config.batch_size * 10
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

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        f1 = 0.
        em = 0.

        ## === take sample ===
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
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

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
        #train_dataset['original_contexts'] = dataset['original_contexts']
        #train_dataset['answers'] = dataset['answers']

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
            batch_order = sorted(np.random.choice(range(n_batches), size = n_batches, replace = False))
            for batch_id in batch_order:
                data_batch = data_batches[batch_id]
                n_samples_in_batch = len(data_batch['start_labels'])
                _, batch_loss = self.optimize(session, data_batch)
                loss += batch_loss * n_samples_in_batch
            loss /= input_size
            
            ## evaluate validation loss
            valid_loss = self.validate(session, val_dataset)
            toc = time.time()
            logging.info("====================== Epoch {} took {} (sec) with training loss: {}, validation loss: {}, exit learning_rate: {}".format(epoch + 1, toc - tic, loss, valid_loss, self.config.learning_rate))

            ## learning rate exponential decay
            self.config.learning_rate *= self.config.decay_rate


            eval_freq = self.config.eval_freq
            if epoch % eval_freq == (eval_freq - 1):
                logging.info("==== Evaluating training set ====")
                self.evaluate_answer(session, dataset, log = True)
                logging.info("==== Evaluating validation set ====")
                self.evaluate_answer(session, val_dataset, log = True)

                if save_parameters:
                    save_path = self.saver.save(session, pjoin(train_dir, "model" + str(int(time.time())) + ".ckpt"))
                    logging.info("** Saved parameters to {}".format(save_path))



