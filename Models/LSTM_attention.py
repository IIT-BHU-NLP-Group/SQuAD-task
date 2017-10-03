import sys
import tensorflow as tf
import numpy as np
import pandas as pd

sys.path.append('../utils')
import Data_Utils as du
from get_word_vectors import get_word_vector

# LOAD DATA
data = du.Data(debug_mode = True, percent_debug_data = 10)

# INITIALIZE HYPER-PARAMETERS
lr=0.006
batch_size = 50
embed_size = 300
hidden_size = 100
max_epochs = 100
dropout = 0.9
# early_stopping = 2

# ADD PLACEHOLDERS
document_input_placeholder = tf.placeholder(tf.float32,(None, None, None))
question_input_placeholder = tf.placeholder(tf.float32,(None, None, None))


# MAKE EMBEDDING LOOKUP TABLE
# Here we directly use vectors from spacy

# MAKE MODEL
def paragraph_encoder(num_steps,  ):
    pass


def question_encoder():
    pass



def decoder():
    pass



def epoch(data):
    pass
    for batch in data.minibatch_iter(batch_size):
        para, question = [], []
        for e in batch:
            para.append(e['para_idx'])
            question.append(e['question_idx'])
        para, question = np.array(para).T, np.array(question)    
        
        feed_dict = {document_input_placeholder :  , question_input_placeholder :  }



