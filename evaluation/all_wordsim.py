# -*- coding:utf-8 -*-
import sys
import os
from evaluation.read_write import read_word_vectors
from evaluation.ranking import *
# from read_write import read_word_vectors
# from ranking import *
def computRho(word_vecs,word_sim_dir):
    res = {}
    res_list = []
    sim_files = ['CH-240.txt','CH-297.txt','CH-MC-30.txt','CH-RG-65.txt']
    for i,filename in enumerate(sim_files):
        manual_dict, auto_dict = ({}, {})
        not_found, total_size = (0, 0)
        for line in open(os.path.join(word_sim_dir, filename),'r'):
            line = line.strip().lower()
            word1, word2, val = line.split()
            if word1 in word_vecs and word2 in word_vecs:
                manual_dict[(word1, word2)] = float(val)
                auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
            else:
                not_found += 1

            total_size += 1
        temp = spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))
        res[filename] = "%15.4f" %temp
        res_list.append(temp)
        # print("%6s" % str(i+1), "%20s" % filename, "%15s" % str(total_size), "%15s" % str(not_found), "%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))
    return res,res_list
def dealAllFile():
    word_vec_dir = sys.argv[1]
    word_sim_dir = sys.argv[2]

    word_vec_files = os.listdir(word_vec_dir)
    word_vec_files = { int(f.split(".")[0].split("ep")[1]) : f for f in word_vec_files}

    word_vec_files = sorted(word_vec_files.items(),key=lambda item:item[0],reverse=True)
    
    temp_res = "ep{} {} {} {} {}"
    for wod_vec_simple_name, word_vec_f in word_vec_files:
        word_vec_path = os.path.join(word_vec_dir,word_vec_f)
        word_vec_s = read_word_vectors(word_vec_path,False)
        res,res_list = computRho(word_vec_s,word_sim_dir)
        print(res)
        cur_res = temp_res.format(wod_vec_simple_name,res['CH-240.txt'],res['CH-297.txt'],res['CH-MC-30.txt'],res['CH-RG-65.txt']) 
        print(cur_res)
    # word_vec_files = [os.path.join(word_vec_dir,f) for f in word_vec_files]

    # word_vecs_s = [read_word_vectors(f,False) for f in word_vec_files]

    # for word_vecs in word_vecs_s:
        # res = computRho(word_vecs,word_sim_dir)
        # print(res)



if __name__=='__main__':
    dealAllFile()
    # word_vec_file = sys.argv[1]
    # word_sim_dir = sys.argv[2]
    # word_vecs = read_word_vectors(word_vec_file,False)

    # print('=================================================================================')
    # print("%6s" %"Serial", "%20s" % "Dataset", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho")
    # print('=================================================================================')

    # for i, filename in enumerate(os.listdir(word_sim_dir)):
    #     manual_dict, auto_dict = ({}, {})
    #     not_found, total_size = (0, 0)
    #     for line in open(os.path.join(word_sim_dir, filename),'r'):
    #         line = line.strip().lower()
    #         word1, word2, val = line.split()
    #         if word1 in word_vecs and word2 in word_vecs:
    #             manual_dict[(word1, word2)] = float(val)
    #             auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
    #         else:
    #             not_found += 1

    #         total_size += 1
    #     print("%6s" % str(i+1), "%20s" % filename, "%15s" % str(total_size), "%15s" % str(not_found), "%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))

    # del word_vecs

