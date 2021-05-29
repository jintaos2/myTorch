
import numpy as np
import re
import time

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
width_Q = 10            # self defined Query matrix width
height_K = 30           # self defined Key matrix height

# onehot encoding
# output size: batch_size * max_sequence_length * dictornary size
# mask size: batch_size * max_sequence_length * height_K
def word2onehot(data:list, dictionary:dict, decode = False):
    prefix = [Decode_Start] if decode else []
    suffix = [Decode_Stop] if decode else [Encode_Stop]
    ret = np.zeros((len(data), max_seq_len, len(dictionary)), dtype='float64')
    mask = np.zeros((len(data), max_seq_len, height_K), dtype='float64')
    for i in range(len(data)):
        words = prefix + re.split(r'[\s]+',data[i].strip()) + suffix
        mask[i,len(words):,:] = - 1e-30
        words += (max_seq_len - len(words)) * suffix
        for j in range(len(words)):
            word_idx = dictionary[words[j]]
            ret[i,j,word_idx] = 1.0
    return ret, mask
    
X, maskX = word2onehot(languageA, dictA, decode = False)
Y, maskY = word2onehot(languageB, dictB, decode = True) 

# print(Y)

def positional_encoding(seq_len, word_vector_len):
    ret = np.zeros((seq_len, word_vector_len))
    for pos in range(seq_len):
        for i in range(word_vector_len):
            if i % 2:
                ret[pos,i] = np.cos(pos/np.power(10000, (i-1)/word_vector_len))
            else:
                ret[pos,i] = np.sin(pos/np.power(10000, i/word_vector_len))
    return ret 

import mytorch as nn

nn.setLogger('./')
nn.Logger.info("xxxxx")
# class Tranformer(nn.Graph):
#     def __init__(self):
#         super().__init__()
#         self.input = nn.Variable(self)
#         self.position_encoding_mat = positional_encoding(max_seq_len, word_vector_len)
#         self.input_embedding = nn.embedding(self,len(dictA), word_vector_len, self.position_encoding_mat)  
        
#         self.output = nn.Variable(self)
#         self.output_embedding = nn.embedding(self,len(dictB), word_vector_len, self.position_encoding_mat)
        
        
#         self.loss = nn.loss_MSE(self)
#         self.optimizer = nn.optim_simple(0.01)
        
#     def forward(self,x,y):
#         self.input.connect(x)
#         self.output.connect(y)
#         self.input_embedding.connect(self.input)
#         self.output_embedding.connect(self.output)
        
#         self.loss.connect(self.input_embedding, self.output_embedding)
        


# model = Tranformer()
# t_start = time.time()
# for i in range(10000):
#     model.forward(X,Y)
#     model.backward()
#     model.step()
#     if np.isnan(model.loss.outputs):
#         break
#     if i % 100 == 99:
#         print("iteration =",i,"\t\tloss =",model.loss.outputs)

# print("training time:",time.time()-t_start)

# print(model.input_embedding.parameters[0])







