
import sys
import os
import time
import re
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import mytorch as nn
nn.setLogger('./')

languageA = ["ich mochte ein bier", "ich mochte ein cola", "ich mochte ein orangensaft"]
languageB = ["i want a beer", "i want a coke", "i want a orange juice"]

Encode_Stop = "\u0000"
Decode_Start = "\u0001"
Decode_Stop = "\u0002"

dictA = {Encode_Stop:0}
dictB = {Decode_Start:0, Decode_Stop:0}

# get dictionary of language A and B
def generate_dict(data:list, target:dict):
    for i in data:
        words = re.split(r'[\s]+',i.strip())
        for j in words:
            target[j] = 0
    idx = 0 
    for i in sorted(target.keys()):
        target[i] = idx 
        idx += 1
    
generate_dict(languageA, dictA) 
generate_dict(languageB, dictB) 


max_seq_len = 20        # pre defined  max sequence length
word_vector_len = 16    # self defined embedding size: 16
width_QK = 10            # self defined Query & Key matrix width

# onehot encoding
# output size: batch_size * max_sequence_length * dictornary size
# mask size: batch_size * max_sequence_length * max_sequence_length
# not masked size: batch_size * max_sequence_length * real_sequence_length
def word2onehot(data:list, dictionary:dict, decode = False):
    prefix = [Decode_Start] if decode else []
    suffix = [Decode_Stop] if decode else [Encode_Stop]
    ret = np.zeros((len(data), max_seq_len, len(dictionary)), dtype='float64')
    mask = np.zeros((len(data), max_seq_len, max_seq_len), dtype='float64')
    for i in range(len(data)):
        words = prefix + re.split(r'[\s]+',data[i].strip()) + suffix
        mask[i,:, len(words):] = - 1e-30
        words += (max_seq_len - len(words)) * suffix
        for j in range(len(words)):
            word_idx = dictionary[words[j]]
            ret[i,j,word_idx] = 1.0
    return ret, mask
    
# batch_size * max_sequence_length * dictA size
# batch_size * max_sequence_length * dictB size
X, maskX = word2onehot(languageA, dictA, decode = False) 
Y, maskY = word2onehot(languageB, dictB, decode = True) 



def positional_encoding(seq_len, word_vector_len):
    ret = np.zeros((seq_len, word_vector_len))
    for pos in range(seq_len):
        for i in range(word_vector_len):
            if i % 2:
                ret[pos,i] = np.cos(pos/np.power(10000, (i-1)/word_vector_len))
            else:
                ret[pos,i] = np.sin(pos/np.power(10000, i/word_vector_len))
    return ret 
position_mat = positional_encoding(max_seq_len, word_vector_len)


class Tranformer(nn.Graph):
    def __init__(self):
        super().__init__()
        
        self.encoder_in: nn.Port = nn.Port(np.zeros((1,max_seq_len, len(dictA))))
        self.decoder_in: nn.Port = nn.Port(np.zeros((1,max_seq_len, len(dictB))))
        
        # use same seq_len and word_vec_len for language A and B
        self.encoder_embedding = nn.embedding(self, word_vector_len, position_mat, self.encoder_in)  
        self.decoder_embedding = nn.embedding(self, word_vector_len, position_mat, self.decoder_in) 
        
        
        self.loss = nn.loss_MSE(self, self.encoder_embedding.outputs[0], self.decoder_embedding.outputs[0])
        self.optimizer = nn.optim_simple(0.01)
        
    def feed(self, x:np.ndarray, y:np.ndarray):
        self.encoder_in.value = x 
        self.decoder_in.value = y
        


        
model = Tranformer()

t_start = time.time()
for i in range(10000):
    
    model.feed(X,Y)
    model.forward()
    model.backward()
    model.step()
    loss = model.loss.outputs[0].extra
    if i % 100 == 99:
        nn.Logger.info(f"iteration = {i}\t\tloss={loss}" )

nn.Logger.info("training time: {}".format(time.time()-t_start))








