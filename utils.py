from data_reader import DataReader
import pickle as pkl

def id2word(id_sentence,id2word_dict):
    sentene = [ id2word_dict[id] for id in id_sentence]
    return ','.join(sentene)

def pkl_load(f_path):
    with open(f_path,'rb') as f:
        return pkl.load(f)
def pkl_dump(data,f_path):
    pkl.dump(data,open(f_path,'wb'))