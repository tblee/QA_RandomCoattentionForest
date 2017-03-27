# Random Coattention Forest for Question Answering

This is a course project for Stanford's CS224N *Natural Language Processing with Deep Learning*. Course website can be found [here](http://web.stanford.edu/class/cs224n/). To run this project, a working installlation of Python 2.7 and TensorFlow 0.12.1 is required. All dependencies are listed in `code/requirements.txt`.

# Data

The question answering dataset is from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/). Downloaded and partitioned dataset can be found in `data/squad`. Word embeddings are obtained from [GloVe](https://nlp.stanford.edu/projects/glove/), `data/squad/glove.trimmed.100.npz` contains the word embeddings used in our model.

# Train and run the model

Train the model by running:

`$ python code/train.py`

When not explicitly specified, the model trains with 3 independent encoder-decoder pairs.

Generate predicted answers for the development set by running:

`$ python code/qa_answer.py`

Note that when the trained model has other than 3 encoder-decoder pairs, `--npairs` must be provided to geneate predictions from proper number of encder-decoder pairs.

The generated predictions will be stored in a JSON file. To evaluate the results, run:

`$ python code/evaluate.py [True dataset location] [Generated predictions]`

`evaluate.py` is the official evaluation script provided by SQuAD.
