# from data_reader import DataReader
import pickle as pkl
import os
import random
# 生成一定范围内指定数目的无重复随机数
def random_without_same(mi, ma, num):
	temp = list(range(mi, ma))
	random.shuffle(temp)
	return temp[0:num]
def id2word(id_sentence,id2word_dict):
    sentene = [ id2word_dict[id] for id in id_sentence]
    return ','.join(sentene)

def pkl_load(f_path):
    with open(f_path,'rb') as f:
        return pkl.load(f)
def pkl_dump(data,f_path):
    pkl.dump(data,open(f_path,'wb'))


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    indices = random_without_same(0,5,5)
    print(indices)