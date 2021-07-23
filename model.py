import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary
import numpy as np
import time

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class VCWEModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, wordid2charid, char_size,noise_dist):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.wordid2charid = wordid2charid # 词-字-id
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.noise_dist = noise_dist.to(self.device)
        # 对应Skip-gram
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        self.char_embeddings = nn.Embedding(char_size, emb_dimension, sparse=False)

        self.cnn_model = CNNModel(32,32,self.emb_dimension).to(self.device)
        self.lstm_model = LSTMModel(self.emb_dimension).to(self.device)
        # print("#------------------------------------------------------------------------------")
        # print(self.cnn_model)
        # print("#------------------------------------------------------------------------------")
        # print(self.lstm_model)     

        initrange = 1.0 / self.emb_dimension # 初始化范围
        # 初始化权重向量
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.v_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.char_embeddings.weight.data, -initrange, initrange)
    
    def forward_noise_strong(self,input_vectors,output_vectors):
        batch,words,embed = output_vectors.shape
        
        noise_words = torch.multinomial(
            self.noise_dist,
            batch * words * batch,
            replacement = True
        ).to(self.device)

        
        noise_words = noise_words.reshape(batch,words,batch)
        # [batch,words,canda_words,embed], conda_words = batch
        noise_vectors = self.v_embeddings(noise_words)

        
        # 修改为适合相似度计算的情况
        output_vectors = output_vectors.view(batch,words,1,embed)

        cos_sim = torch.cosine_similarity(output_vectors,noise_vectors,dim=3)

        
        # TODO 这里最好和输出目标词的数量一样
        strong_neg_sample = torch.zeros(batch,words,dtype=torch.long)
        strong_neg_embed = torch.zeros(batch,words,self.emb_dimension)

        # TODO 取前k个负样本，这取1，是因为前面计算是按照输出词的样本计算的，后面可以改
        values,indices = cos_sim.topk(k=1,dim=2,largest=True,sorted=True)
        
        for i,f in enumerate(indices):
            strong_neg_sample[i] = noise_words[i,range(f.shape[0]),f.squeeze()]
            strong_neg_embed[i] = noise_vectors[i,range(f.shape[0]),f.squeeze()]
        del noise_vectors
        del noise_words
        del output_vectors
        del cos_sim

        return strong_neg_sample.to(self.device), strong_neg_embed.to(self.device)


    def forward_noise_strong_avg(self,avg_output_vectors,neg_num=5):
        '''在glyce中是没有使用求平均，在vcwe中使用使用了求平均'''
        batch,embed = avg_output_vectors.shape
        # time1 = time.time()
        avg_output_vectors = avg_output_vectors.view(batch,1,embed)
        miniBatch = batch //2
        noise_words = torch.multinomial(
            self.noise_dist,
            batch * 1 * miniBatch, # 由于这里是平均过得，为batch个词采样，batch个候选词
            replacement = True
        )
        # time2 = time.time()
        noise_words = noise_words.reshape(batch,miniBatch)
        noise_vectors = self.v_embeddings(noise_words).view(batch,miniBatch,embed)

        
        # time3 = time.time()
        cos_sim = torch.cosine_similarity(avg_output_vectors,noise_vectors,dim=2) # [batch,miniBatch],每个batch有batch个候选项    
        # time4 = time.time()
        # strong_neg_sample = torch.zeros(batch,neg_num,dtype=torch.int).to(self.device)
        # strong_neg_embed = torch.zeros(batch,neg_num,embed).to(self.device)

        values,indices = cos_sim.topk(k=neg_num,dim=1,largest=True,sorted=True)
        indices_q = indices.unsqueeze(2).repeat(1,1,embed)
        # print(noise_words.shape)
        # print(noise_vectors.shape)
        # print(indices.shape)
        # print(indices_q.shape)
        # # print(indices)
        # indices_q = indices.repeat(1,embed)
        # print(indices_q.shape)
        # print(indices_q)
        
        
        # print(fuck.shape)
        # print(fuck_example.shape)
        # print(fuck.shape)
        # time5 = time.time()
        # TODO 以前的方式使用循环来获得数据
        # for i,f in enumerate(indices):
        #     strong_neg_sample[i] = noise_words[i,f]
        #     strong_neg_embed[i] = noise_vectors[i,f]
        # time6 = time.time()

        # TODO 现在的方式，使用gather来获得数据
        strong_neg_embed = torch.gather(noise_vectors,1,indices_q)
        strong_neg_sample = torch.gather(noise_words,1,indices)
        # print(fuck_example[0])
        # print(strong_neg_sample[0])
        # TODO 这里得到的是一样的
        # print(fuck[0][0]) 
        # print(strong_neg_embed[0][0])
        # print(time2-time1,time3-time2,time4-time3,time5-time4,time6-time5)
        return strong_neg_sample,strong_neg_embed




    def forward(self, pos_u, pos_v, neg_v, img_data):
        img_emb = self.cnn_model.forward(img_data) # [5031, 100]
        
        # time1 = time.time()
        # print("img_emb's shape",img_emb.shape)
        emb_u = self.u_embeddings(pos_u)     # 中心词 
        emb_vv = self.v_embeddings(pos_v)
        emb_v = emb_vv.mean(dim=1) # 范围词,
        # time2 = time.time()
        # emb_neg_v = self.v_embeddings(neg_v) # 负样本词 [128 5 100]

        

        # print("emb_u's {} emb_v's {} emb_neg_v's {}".format(emb_u.shape,emb_v.shape,emb_neg_v.shape))


        # strong_neg_sample, strong_neg_embed = self.forward_noise_strong(emb_u,emb_vv)
        # TODO StrongNeg
        neg_v, emb_neg_v = self.forward_noise_strong_avg(emb_v)

        # time3 = time.time()
        # print("strong_neg_sample's {} strong_neg_embed's {}".format(strong_neg_sample.shape,strong_neg_embed.shape))
        # print("avg_strong_neg_sample's {} avg_strong_neg_embed's {}".format(avg_strong_neg_sample.shape,avg_strong_neg_embed.shape))
        # print("--")
        # print("pos_v's shape",pos_v.shape) 
        pos_v=pos_v.view(-1).cpu() # [4096, 9] [line_batch_size*sample_batch_size, window_size*2-1]
        # print("pos_v's shape",pos_v.shape) # [18]
        temp = self.wordid2charid[pos_v].reshape(1,-1) # 18 * maxwordlength = 90
        # print("temp's shape",temp.shape)
        temp = torch.from_numpy(temp).to(self.device).long() # [1, 184320] ([1, 90])
        # print("temp's shape",temp.shape)

        # print("chosen emb",img_emb[temp.reshape(1, -1)].shape) # [1, 90, 100]
        # print(temp)
        # 这种是所有字符在一起训练，然后通过temp来索引出对应的图像Emb
        lstm_input = img_emb[temp.reshape(1, -1)].view(len(pos_v), -1, self.emb_dimension)  # ([18, 5, 100])
        # print("lstm's input shape",lstm_input.shape)
        del temp
        lstm_input = torch.transpose(lstm_input, 0, 1)          #  self.data.maxwordlength, batch_size, embedding_dim
        emb_char_v = self.lstm_model.forward(lstm_input, len(pos_v))
        emb_char_v = emb_char_v.view(pos_u.size(0),-1,self.emb_dimension)
        emb_char_v = torch.mean(emb_char_v,dim=1)

        # 这里的操作和刚才一样
        pos_neg_v=neg_v.view(-1).cpu()
        temp = self.wordid2charid[pos_neg_v].reshape(1,-1)
        temp = torch.from_numpy(temp).to(self.device).long()
        lstm_input2 = img_emb[temp.reshape(1, -1)].view(len(pos_neg_v), -1, self.emb_dimension)
        del temp
        lstm_input2 = torch.transpose(lstm_input2, 0, 1)
        emb_neg_char_v = self.lstm_model.forward(lstm_input2, len(pos_neg_v))
        emb_neg_char_v = emb_neg_char_v.view(pos_u.size(0),-1,self.emb_dimension)

        # 中心词与范围词，既正样本词
        c_score = torch.sum(torch.mul(emb_u, emb_char_v), dim=1)
        c_score = torch.clamp(c_score, max=10, min=-10)
        c_score = -F.logsigmoid(c_score)

        # 中心词与负样本词，既负样本词
        neg_c_score = torch.bmm(emb_neg_char_v, emb_u.unsqueeze(2)).squeeze()
        neg_c_score = torch.clamp(neg_c_score, max=10, min=-10)
        neg_c_score = -torch.sum(F.logsigmoid(-neg_c_score), dim=1)

        # 中心词与范围词的图像emb，既正样本词
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        # 中心词与负样本的词的图像emb，既负样本词
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        # time4 = time.time()

        # print(time2-time1,time3-time2,time4-time3)
        return torch.mean(c_score + neg_c_score + score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w', encoding="utf-8") as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))


class LSTMModel(nn.Module):

    def __init__(self, emb_dimension, d_a=128):
        super().__init__()
        self.emb_dimension = emb_dimension
        self.hidden_dim = emb_dimension
        self.lstm = nn.LSTM(input_size=self.emb_dimension, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True)
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.lstm.all_weights[0][0], -initrange, initrange)
        init.uniform_(self.lstm.all_weights[0][1], -initrange, initrange)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.linear_first = torch.nn.Linear(2*self.hidden_dim, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, 1)
        self.linear_second.bias.data.fill_(0)
        self.linear_third = torch.nn.Linear(2*self.hidden_dim, self.emb_dimension)
        self.linear_third.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        # first is the hidden h
        # second is the cell c
        return (torch.zeros(2, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(2, batch_size, self.hidden_dim).to(self.device))

    def forward(self, input, batch_size):                 
        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(input, self.hidden)             
        a=self.linear_first(lstm_out)             
        a=torch.tanh(a)
        a=self.linear_second(a)           
        a=F.softmax(a,dim=0)             
        a=a.expand(5,batch_size,2*self.hidden_dim)
        y=(a*lstm_out).sum(dim=0)    
        y=self.linear_third(y)  
        return y

class CNNModel(nn.Module):
    def __init__(self, output1, output2, emb_dimension):    #output1=32 output2=32
        super().__init__()
        self.emb_dimension = emb_dimension
        self.conv1 = nn.Conv2d(1, output1, (3, 3))
        self.conv2 = nn.Conv2d(output1, output2, (3, 3))
        self.hidden2result = nn.Linear(output2*64, emb_dimension)    
        self.bn1 = nn.BatchNorm2d(output1)
        self.bn2 = nn.BatchNorm2d(output2)
        self.bn3 = nn.BatchNorm1d(emb_dimension)
        initrange = 1.0 / self.emb_dimension
        initrange1 = 1e-4
        initrange2 = 1e-2
        init.uniform_(self.conv1.weight.data, -initrange1, initrange1)
        init.uniform_(self.conv2.weight.data, -initrange2, initrange2)
        init.uniform_(self.hidden2result.weight.data, -initrange, initrange)

    def forward(self, x):  
        x = self.conv1(x) # [5031, 32, 38, 38]
        # print("x's shape",x.shape)
        x = F.max_pool2d(self.bn1(x), 2)
        x = self.conv2(x) #  [5031, 32, 17, 17]
        # print("x's shape",x.shape)
        x = F.max_pool2d(self.bn2(x), 2)
        x = x.view(x.size()[0], -1) # [5031, 2048]
        # print("x's shape",x.shape)
        x = self.hidden2result(x)
        # print("x's shape",x.shape) # [5031, 100]
        x = F.relu(self.bn3(x))
        return x                              

if __name__ == "__main__":
    # test_model = VCWEModel
    pass