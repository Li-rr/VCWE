from data_reader import DataReader

def id2word(id_sentence,id2word_dict):
    sentene = [ id2word_dict[id] for id in id_sentence]
    return ','.join(sentene)