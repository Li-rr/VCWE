import numpy as np
import torch
from torch.utils.data import Dataset
import random
import sys
import utils

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, vocabulary_file, char2ix_file, maxwordlength, discard, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.discard = discard
        # data = np.load(char2ix_file,allow_pickle=True)
        # print(data)
        # self.char2id = data['char_to_ix'].item()

        self.char2id = utils.pkl_load(char2ix_file)
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.maxwordlength = maxwordlength  

        self.word2id = dict()
        self.id2word = dict()
        self.wordid2charid = []
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName # zh_wiki
        self.vocabulary_file = vocabulary_file
        self.char2ix_file = char2ix_file
        self.read_words()
        self.initTableNegatives()
        self.initTableDiscards()

        self.noise_dist = None

        self.getNoiseDist()
    

    def read_words(self):
        # 读取词表,这里获取所有词的词表
        with open(self.vocabulary_file,"r",encoding="utf-8") as f:
            s=f.readline().strip().split() # 读取第一行
            print(s)
            self.sentences_count=int(s[0]) # 获取句子的数量
            self.token_count=int(s[1]) # 获取词的数量
            s = f.readline().strip().split() 
            print(s)
            wid = 0
            while s:
                w=s[0] # 获得词
                c=int(s[1]) # 获得词的统计
                # print(w,c)
                # break
                self.word2id[w] = wid # word-id
                self.id2word[wid] = w # id-word
                self.word_frequency[wid] = c # 词频
                wid += 1
                s = f.readline().strip().split() # 读取下一行
        # 读取词表完成
        # sys.exit(1)
        for i in range(len(self.id2word)):
            word = self.id2word[i] # 获取词
            w=[]
            for j in range(len(word)): # 获取词中的字
                try:
                    w.append(self.char2id[word[j]])
                except:
                    w.append(0)
            while len(w)<self.maxwordlength:w.append(0) # 默认为 5
            w=w[:self.maxwordlength]
            self.wordid2charid.append(w) # 得到词中的字
            
        self.wordid2charid=np.array(self.wordid2charid)
        print("Total embeddings: " + str(len(self.word2id)))

    def getNoiseDist(self):
        # 获得噪声分布
        print(self.token_count)
        # print(len(self.word_frequency))
        # print(self.word_frequency.sum())
        n_vocab = len(self.word_frequency)

        word_freqs = {id: t/n_vocab for id,t in self.word_frequency.items()}
        word_freqs = np.array(list(word_freqs.values()))

        unigram_dist = word_freqs/ word_freqs.sum()
        # noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

        self.noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
        # print(len(word_freqs))
        # print(word_freqs)
        # print(noise_dist)
        pass

    def initTableDiscards(self):
        t = self.discard                       #t is a chosen threshold, typically around 10^−5
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f)


    def initTableNegatives(self):
        '''初始化负采样表,这里可以不要'''
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class Word2vecDataset(Dataset):
    def __init__(self, data, window_size, sample_batch_size, neg_num):
        self.data = data
        self.window_size = window_size
        self.sample_batch_size = sample_batch_size
        self.neg_num = neg_num
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    # 如果在词表中
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = self.window_size

                    result=[]
                    for i, u in enumerate(word_ids):
                        if i>boundary and i+boundary<len(word_ids):
                            v_list=[]
                            neg=self.data.getNegatives(u, self.neg_num)
                            for j, v in enumerate(word_ids[max(i - boundary, 0):i + boundary]):
                                if j!=boundary:
                                    v_list.append(v)
                            v_np=np.array(v_list)
                            result.append((u,v_np,neg,len(self.data.id2word[u])))
                    random.shuffle(result)
                    result=result[:self.sample_batch_size]
                    result.sort(key=lambda x: x[3],reverse=True)
                    return result

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _, _  in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _, _  in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v, _ in batch if len(batch) > 0]
        all_length = [length for batch in batches for _, _, _, length in batch if len(batch) > 0]
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v), torch.LongTensor(all_length)
