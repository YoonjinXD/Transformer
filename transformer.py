import os
import os.path
import math
import json
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

import nltk
import random
from collections import Counter
import pickle
import pdb
import argparse
import seaborn as sns

dtype = torch.FloatTensor

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
if torch.cuda.is_available():
    torch.cuda.init()
    dtype = torch.cuda.FloatTensor
    print("Cuda is availiable... Using %d devices" %(torch.cuda.device_count()))

class Configuration(object):
    def __init__(self):
        self.d_model= 512
        self.FF_innerlayer_dim= 1024
        self.key_vector_dim= 256
        self.value_vector_dim= 256
        self.emb_dimension= 512
        self.encoder_layer_num= 6
        self.decoder_layer_num= 6
        self.attention_num_heads= 8
        self.batch_size = 32
config = Configuration()
print("CONFIG:", vars(config))        
"""
    [Multi Head Self Attentional Module]
     Where B= batch_size, S= sequence_size, D= model_dimeision, H= head_num
     - Input= (B, S, D)
     - Q and K vector= (B, S, d_k*H) / V vector= (B, S, d_v*H)
     - Attention= (B, H, S, S)
     - Context= (B, H, S, d_v)
     - Output= (B, S, D) 
"""
class Multihead_SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.batch_size = config.batch_size
        self.d_k = config.key_vector_dim
        self.d_v = config.value_vector_dim
        self.h = config.attention_num_heads
        
        self.W_q = nn.Linear(self.d_model, self.d_k*self.h)
        self.W_k = nn.Linear(self.d_model, self.d_k*self.h)
        self.W_v = nn.Linear(self.d_model, self.d_v*self.h)
        self.W_o = nn.Linear(self.d_v*self.h, self.d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.norm =  nn.LayerNorm(self.d_model)
        # (OPTIONAL) for visualization. TODO: will be deleted later
        # self.scores = None
        
    def scaled_dot_product(self, q, k, v, attn_mask):
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.size(-1))
        
        if attn_mask is not None: scores = scores.masked_fill(attn_mask, -1e9)
        
        attention = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attention, v)
        return context, attention
        
    
    def forward(self, q, k, v, attn_mask):
        residual = q
        q, k, v= self.W_q(q), self.W_k(k), self.W_v(v)
        # (1) Split q, k, v vectors from (B, S, k(v)_dim*H) to (B, H, S, k(v)_dim) and create the corresponding attention masks
        # q_s = (B, H, S, k_dim)
        # k_s = (B, H, S, k_dim)
        # v_s = (B, H, S, v_dim))
        # attn_mask : (B, H, d_k, d_v)
        q_s = q.view(self.batch_size, -1, self.h, self.d_k).transpose(1,2)
        k_s = k.view(self.batch_size, -1, self.h, self.d_k).transpose(1,2)
        v_s = v.view(self.batch_size, -1, self.h, self.d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.h, 1, 1)
        
        # (2) Head_i = Scaled_Dot_Product(q*W^q_i, k*W^k_i, v*W^v_i)
        context, attention = self.scaled_dot_product(q_s, k_s, v_s, attn_mask)
        # (3) Concat context vectors and Resize by using the W_o(Output weight)
        concated = context.transpose(1, 2).contiguous().view(self.batch_size, -1, self.h*self.d_v)
        output = Variable(self.W_o(concated)).cuda()
        return self.norm(output+residual), attention
        
        
"""
[Position-wise Feed-Forward Networks]
Either fc linear and conv with kernel size=1 can be used.
TODO: In fact, kernel size can be extended. Find a proper kernel size in the data.
  - Input= (B, S, D)
  - Inner_State =(B, S, d_ff)
  - Output = (B, S, D)
"""
class Poswise_FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_ff = config.FF_innerlayer_dim
        self.d_model = config.d_model
        self.conv1 = nn.Conv1d(self.d_model, self.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(self.d_ff, self.d_model, kernel_size=1)
        self.norm = nn.LayerNorm(self.d_model)       
 
    def forward(self, x):
        residual = x
        output = nn.ReLU()(self.conv1(x.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.norm(output + residual)

"""[Attention mask modules]"""
def get_attn_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    
    # eq(zero) is PAD token
    # batch_size x 1 x len_k(=len_q), one is masking
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # batch_size x len_q x len_k
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_subsequent_attn_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


"""[Positional Encoding]"""
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

"""[Modules]"""
class EncoderBlock(nn.Module):
    """[Encoder Block]"""
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.attention_layer = Multihead_SelfAttention(config)
        self.feedforward_layer = Poswise_FeedForward(config)
        #self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs, attn_mask):
        attn_mask = Variable(attn_mask).cuda()
        outputs, attention = self.attention_layer(inputs, inputs, inputs, attn_mask)
        encoder_outputs = self.feedforward_layer(outputs)
        return encoder_outputs, attention

class DecoderBlock(nn.Module):
    """[Decoder Block]"""
    def __init__(self, config):
        super().__init__()
        self.masked_attention_layer = Multihead_SelfAttention(config)
        self.attention_layer = Multihead_SelfAttention(config)
        self.feedforward_layer = Poswise_FeedForward(config)

    def forward(self, inputs, encoder_outputs, masked_attn_mask, attn_mask):
        outputs, dec_self_attnetion = self.masked_attention_layer(inputs, inputs, inputs, masked_attn_mask)
        outputs, dec_enc_attention = self.attention_layer(outputs, encoder_outputs, encoder_outputs, attn_mask)
        decoder_outputs = self.feedforward_layer(outputs)
        return decoder_outputs, dec_self_attnetion, dec_enc_attention


class Encoder(nn.Module):
    def __init__(self, config, input_vocab_size, input_len):
        super().__init__()
        d_emb = config.emb_dimension
        n_layers = config.encoder_layer_num
        self.position_info = Variable(torch.tensor([i for i in range(100)])).cuda()
        self.input_emb = nn.Embedding(input_vocab_size, d_emb)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(input_len+1, d_emb),freeze=True)
        self.layers = nn.ModuleList([EncoderBlock(config.d_model) for _ in range(n_layers)])

    def forward(self, inputs): # inputs = [B, S]
        encoded_inputs = self.input_emb(inputs) + self.pos_emb(self.position_info)
        attn_mask = get_attn_mask(inputs, inputs)
        attns = []

        for layer in self.layers:
            outputs, attn = layer(encoded_inputs, attn_mask)
            attns.append(attn)

        return outputs, attns

class Decoder(nn.Module):
    def __init__(self, config, target_vocab_size, target_len):
        super().__init__()
        d_emb = config.emb_dimension
        n_layers = config.decoder_layer_num
        self.position_info = torch.tensor([i for i in range(100)])
        self.target_emb = nn.Embedding(target_vocab_size, d_emb)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(target_len+1, d_emb), freeze=True)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(n_layers)])

    def forward(self, targets, enc_inputs, enc_outputs): # targets = [B, S]
        position_info = Variable(self.position_info).cuda()
        encoded_targets = self.target_emb(targets) + self.pos_emb(position_info)
        subsequent_attn_mask = Variable(get_subsequent_attn_mask(targets)).cuda()
        attn_mask = get_attn_mask(targets, targets).byte()

        self_attn_mask = torch.gt((attn_mask + subsequent_attn_mask), 0)
        combo_attn_mask = get_attn_mask(targets, enc_inputs)
        self_attns, combo_attns = [], []

        for layer in self.layers:
            outputs, self_attn, combo_attn = layer(encoded_targets, enc_outputs, self_attn_mask, combo_attn_mask)
            self_attns.append(self_attn)
            combo_attns.append(combo_attn)

        return outputs, self_attns, combo_attns


"""[Full Model]"""
class Transformer(nn.Module):
    def __init__(self, config, source_size, target_size, source_len, target_len):
        super().__init__()
        self.d_model = config.d_model
        self.encoder = Encoder(config, source_size, source_len)
        self.decoder = Decoder(config, target_size, target_len)
        self.flatten_layer = nn.Linear(self.d_model, target_size, bias=False)

    # encoder inputs = torch.tensor([B, S])
    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs, encoder_attns = self.encoder(encoder_inputs)
        decoder_outputs, decoder_self_attns, decoder_combo_attns = self.decoder(decoder_inputs, encoder_inputs, encoder_outputs)
        prob_outputs = self.flatten_layer(decoder_outputs)
        return prob_outputs, encoder_attns, decoder_self_attns, decoder_combo_attns


"""[Data Preprocessing]"""
class Vocabulary(object):
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.w2i:
            self.w2i[word] = self.idx
            self.i2w[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[word]

    def __len__(self):
        return len(self.w2i)

def build_vocab(lang_name, text):
    counter = Counter()
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    corpus = []
    max_len = 0

    for sentence in text:
        words = sentence.lower().split(' ')
        for word in words:
            vocab.add_word(word)
        corpus.append(' '.join(words[:-1]))
        if len(words) > max_len:
            max_len = len(words)

    with open('vocab/%s_vocab.pkl' %(lang_name), 'wb') as f:
        pickle.dump(vocab, f)

    print("%s corpus: %d num of words and %d num of sentences" %(lang_name, len(vocab), len(corpus)))
    return corpus, vocab, max_len

def make_batch(batch_size, corpus, vocab, max_len):
    batch_num = int(len(corpus)/batch_size)
    result = []
    for i in range(batch_num):
        batch = []
        for sentence in corpus[i*batch_size:(i+1)*batch_size]:
            words = sentence.split(' ')
            temp = []
            for word in words:
                if word in vocab.w2i:
                    temp.append(vocab.w2i[word])
                else:
                    temp.append(vocab.w2i['<unk>'])
            for i in range(max_len - len(words)):
                temp.append(vocab.w2i['<pad>'])
            batch.append(temp)
        result.append(torch.tensor(batch))
    return result

def show_clearly(idx_seq, vocab):
    result = [vocab.i2w[n.item()] for n in idx_seq.squeeze()]
    temp = []
    for word in result:
        if word == '<pad>':
            continue
        else:
            temp.append(word)
    return ' '.join(temp).encode('ascii', 'ignore')

def show_attn(attn_name, data_num, attns, input_seq, pred_seq):
    fig = plt.figure(figsize=(10, 10))
    sns.set_style("ticks",{"xtick.major.size":7, "ytick.major.size":7})
    attns = Variable(attns).cpu()
    attn = attns[data_num][0].data.numpy()
    input_seq = input_seq.split(' ')
    pred_seq = pred_seq.split(' ')
    g = sns.heatmap(attn[:len(input_seq), :len(pred_seq)])
    g.set_xticklabels(input_seq, rotation=90)
    g.set_yticklabels(pred_seq, rotation=0)
    g.xaxis.tick_top()
    plt.savefig('results/%s_attn.png' %(attn_name))

"""[Pipeline]"""
def main():
    parser = argparse.ArgumentParser(description='transformer')
    parser.add_argument('mode', metavar='mode', type=str)
    args = parser.parse_args()
    mode = args.mode

    # 1. Load data
    print("Load dataset ...")
    text_en = open('data/translation/train/train.en', 'r', encoding='utf-8').readlines()[:50000]
    text_de = open('data/translation/train/train.de', 'r', encoding='utf-8').readlines()[:50000]

    # 2. Preprocessing
    print("Preprocess ...")
    en_corpus, en_vocab, en_len = build_vocab('English', text_en)
    de_corpus, de_vocab, de_len = build_vocab('Deutsch', text_de)

    print("English max length: %d \nDeutsch max length: %d" %(en_len, de_len))

    # 3. Training Setting
    epochs = 40
    avg_loss = 0
    total_loss = 0
    source_size = len(en_vocab) +1
    target_size = len(de_vocab) +1
    input_len = en_len
    target_len = de_len

    # 4. Create Model
    model = Transformer(config, source_size, target_size, input_len, target_len)
    model.cuda()
    #model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    # test mode Exception
    if(mode == 'test'): 
        epochs = 0
        model.load_state_dict(torch.load('checkpoint.pkl'))

    en_inputs = make_batch(32, en_corpus, en_vocab, en_len)
    de_inputs = make_batch(32, de_corpus, de_vocab, de_len)
    dataset = list(zip(en_inputs, de_inputs))
    total_dataset = len(dataset)

    # 5. Training Iterator
    for epoch in range(epochs):
        optimizer.zero_grad()

        for i, (enc_inputs, dec_inputs) in enumerate(dataset):
            enc_inputs = Variable(enc_inputs).cuda()
            dec_inputs = Variable(dec_inputs).cuda()
            outputs, _, _, _ = model(enc_inputs, dec_inputs)
            outputs = outputs.view(-1, outputs.size(2))
            loss = criterion(outputs, dec_inputs.contiguous().view(-1))
            avg_loss += loss
            total_loss += loss
            loss.backward()
            optimizer.step()

            if ((i+1)%100 == 0):
                print('Epoch[%d/%d]' %(epoch + 1, epochs), 'Step[%02d/%d]' %((i+1), total_dataset), 'loss=', '{:.6f}'.format(avg_loss/100))
                avg_loss = 0

        print(' > %d Epoch Summary:' %(epoch + 1), 'Total avg loss = ', '{:.6f}'.format(total_loss/total_dataset))
        avg_loss = 0
        total_loss = 0

        # save the model every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'checkpoint.pkl')

    # 6. Show sample result
    # Load Test data
    print("Load Test dataset ...")
    text_en = open('data/translation/test/newstest2012.en', 'r', encoding='utf8').readlines()[:1000]
    text_de = open('data/translation/test/newstest2012.de', 'r', encoding='utf8').readlines()[:1000]

    en_corpus, _, _ = build_vocab('English', text_en)
    de_corpus, _, _ = build_vocab('Deutsch', text_de)
    print("English max length: %d \nDeutsch max length: %d" %(en_len, de_len))

    en_inputs = make_batch(32, en_corpus, en_vocab, 100)
    de_inputs = make_batch(32, de_corpus, de_vocab, 100)
    dataset = list(zip(en_inputs, de_inputs))
    total_dataset = len(dataset)
    model.eval()
   
    print('\n--------------<Sample Result>------------------\n')
    enc_inputs, dec_inputs = dataset[0]
    enc_inputs = Variable(enc_inputs).cuda()
    dec_inputs = Variable(dec_inputs).cuda()
    predict, enc_self_attns, dec_self_attns, dec_combo_attns = model(enc_inputs, dec_inputs)

    data_num = 11
    input_seq = str(show_clearly(enc_inputs[data_num], en_vocab))[2:]
    target_seq = str(show_clearly(dec_inputs[data_num], de_vocab))[2:]
    pred_seq = str(show_clearly(predict[data_num].data.max(1, keepdim=True)[1], de_vocab))[2:]
    print('(I)', input_seq, '\n(T)', target_seq,'\n(P)', pred_seq)
    show_attn('enc_self', data_num, enc_self_attns[5], input_seq, pred_seq)
    show_attn('dec_self', data_num, dec_self_attns[5], input_seq, pred_seq)
    show_attn('dec_combo', data_num, dec_combo_attns[5], input_seq, pred_seq)
    print('\nAttention Graphs are created. Check \'results\' directory.')

main()
