from functools import partial
from ray.tune.session import report
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm
import numpy as np
import argparse
import sys,os
from PIL import Image
from scipy import misc
from torchsummary import summary
from visualdl import LogWriter # 百度可视化工具
import datetime

from torch.optim import lr_scheduler
from utils import ensure_dir, id2word,pkl_load,trial_name_string
from data_reader import DataReader, Word2vecDataset
from model import VCWEModel
from optimization import VCWEAdam
from logginger import init_logger 
import random

# 超参数自动搜索
from ray import tune
from ray.tune import CLIReporter, trial
from ray.tune.schedulers import ASHAScheduler,FIFOScheduler

# 导入评估词向量的东西
from evaluation.all_wordsim import computRho


class Word2VecTrainer:
    def __init__(self, input_file, vocabulary_file, img_data_file, char2ix_file, output_dir, maxwordlength, emb_dimension, line_batch_size, sample_batch_size, 
                neg_num, window_size, discard, epochs, initial_lr, seed,exp_name):

        self.input_file = input_file
        self.vocabulary_file = vocabulary_file
        self.seed = seed
        self.img_data_file = img_data_file

        self.char2ix_file = char2ix_file
        self.maxwordlength = maxwordlength
        self.discard = discard

        self.window_size = window_size
        self.sample_batch_size = sample_batch_size
        self.neg_num = neg_num
        self.line_batch_size = line_batch_size

        self.output_dir = output_dir
        self.emb_dimension = emb_dimension
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.exp_name = exp_name



    def rho_acc(self,embds,word2id):
        word_vecs = {}

        for wid,w in word2id.items():
            word_vecs[w] = embds[wid]

        _,res_list = computRho(word_vecs,'/home/stu/LRR/trainW2C/VCWE/evaluation/word-sim/')

        # print(res_list)
        all_rho = sum(res_list)
        # print(all_rho)
        return all_rho,res_list

    def train_tune(self,num_samples):
      
        # 配置超参数搜索空间
        hyper_para_conf = {
            # 'lr': tune.loguniform(1e-4, 1e-1),
            'seed': tune.randint(1000,9000),
            'lr': tune.grid_search([1e-3,2e-3,5e-4]),
            'emb_dim': tune.grid_search([100,128]) # 100，128作为格点
        }
        # # 配置调度器
        # scheduler = ASHAScheduler(
        #     metric="rho",
        #     mode="max",
        #     max_t=self.epochs,
        #     grace_period=1,
        #     reduction_factor=2
        # )
        # scheduler = FIFOScheduler(
            
        # )
        # 在命令行打印实验报告
        report = CLIReporter(
            metric_columns=['rho','loss','training_iteration','ch240','ch297','mc30','rg65']
        )
        result = tune.run(
            self.train,
            name = self.exp_name, # 实验名称
            local_dir= "./ray_results",
            trial_name_creator= tune.function(partial(trial_name_string,trail_name=self.exp_name,mode="name")),
            trial_dirname_creator  = tune.function(partial(trial_name_string,trail_name=self.exp_name,mode="dir")),
            # 指定训练资源
            resources_per_trial={"cpu": 2, "gpu": 1},
            config=hyper_para_conf, # 超参数空间
            num_samples=num_samples, # 可以看作实验次数
            # scheduler=scheduler, # 调度器
            progress_reporter=report, # 进度报告器

        )

    def train(self,config):
        '''
        在这里改造训练部分的代码
        '''
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        # self.img_data = np.load(img_data_file)
        self.img_data = pkl_load(self.img_data_file)
        self.data = DataReader(self.input_file, self.vocabulary_file, self.char2ix_file, self.maxwordlength, self.discard, self.seed)
        # sample_batch_size，用于构建基本的数据集
        dataset = Word2vecDataset(self.data, self.window_size, self.sample_batch_size, self.neg_num)
        # line_batch_size是一次取几行数据，可以设为1
        self.dataloader = DataLoader(dataset, batch_size=self.line_batch_size,
                                     shuffle=True, num_workers=0, collate_fn=dataset.collate)

        self.output_dir = self.output_dir
        self.emb_size = len(self.data.word2id)
        self.char_size = len(self.data.char2id)+1       #5031

        

        


        self.use_cuda = True #torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 读取图片数据 img ([5031, 1, 40, 40]) 第一个维度代表不同的字
        self.img_data = torch.from_numpy(self.img_data).to(self.device)

        if self.img_data.dim() == 3:
            print(self.img_data.shape)
            self.img_data = self.img_data.unsqueeze(1)
            print(self.img_data.shape)
        
        

        self.num_train_steps= int(len(self.dataloader) * self.epochs)

        print("--------------------------------初始化完成")

        #----------------------------------------------------------------------
        trial_name = tune.get_trial_name() # 实验代号
        exp_dir = os.path.join("/home/stu/LRR/trainW2C/VCWE/logs/",self.exp_name) # 当前实验的文件夹
        ensure_dir(exp_dir)
        logs_path = os.path.join(exp_dir,trial_name)
        # 日志文件的格式为: ./logs/新场/新场_001
        # logs_path = "./logs/" + self.exp_name +cur_time        


        emb_dir = "/home/stu/LRR/trainW2C/VCWE/embedding/{}/{}/".format(self.exp_name,trial_name)
        ensure_dir(emb_dir)
        writer = LogWriter(logdir=logs_path)
        self.loggger = init_logger("myVcwe_{}".format(self.exp_name),logs_path)
        #---------------------------------------------
        VCWE_model = VCWEModel(
            self.emb_size,
            config['emb_dim'],
            self.data.wordid2charid, 
            self.char_size,
            self.data.noise_dist,
            self.loggger,
            self.exp_name
        )
        if self.use_cuda:
            VCWE_model.cuda()
        print(VCWE_model)
        no_decay = ['bias']
        optimizer_parameters = [
             {'params': [p for n, p in VCWE_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
             {'params': [p for n, p in VCWE_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
             ]
		
        print("lr:",config["lr"])
        print("seed:",config["seed"])
        print("num_train_steps=",self.num_train_steps)
        optimizer = VCWEAdam(optimizer_parameters,
                             lr=config['lr'],
                             warmup=0.1,
                             t_total=self.num_train_steps)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=1e-4, last_epoch=-1)
        steps = 0
        accumulation_steps = 4 # 用于梯度积累
        for epoch in range(self.epochs):

            self.loggger.info("Epoch: " + str(epoch + 1))
            '''
            进来一个batch的数据，计算一次梯度，更新一次网络
            1. 获取Loss
            2. optimizer.zero_grad()清空过往梯度；
            3. loss.backward()反向传播，计算当前梯度；
            4. optimizer.step()根据梯度更新网络参数

            TODO 梯度累加策略
            1. 获取loss
            2. loss.backward() 反向传播，计算当前梯度；
            3. 多次循环步骤1-2，不清空梯度，使梯度累加在已有梯度上；
            4. 梯度累加了一定次数后，先 optimizer.step() 
                根据累计的梯度更新网络参数，然后 optimizer.zero_grad() 清空过往梯度，
                为下一波梯度累加做准备；
            '''


            running_loss = 0.0
            epoch_steps = 0
            for i, sample_batched in enumerate(self.dataloader):
                # print('len(sample_batched) -->',len(sample_batched))
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device) # [4096] 中心词，pos_u在这里是词
                    pos_v = sample_batched[1].to(self.device) # [4096, 9] 范围词
                    neg_v = sample_batched[2].to(self.device) # [4906, 5] 负样本词


                    # lengths = sample_batched[3].to(self.device)
                    # 查看范围词
                    # sentence = id2word(id_sentence=pos_v[0].cpu().numpy(),id2word_dict=self.data.id2word)
                    # print("pos_u's {} pos_v's {} neg_v's {}".format(pos_u.shape,pos_v.shape,neg_v.shape))
                    # print(pos_v)
                    # print(sentence)
                    # optimizer.zero_grad()
                    loss = VCWE_model.forward(pos_u, pos_v, neg_v, self.img_data)
                    # print("loss's type",type(loss),loss)
                    running_loss += loss.item()
                    loss_2 = loss.item()
                    loss = loss / accumulation_steps  # 2.1 loss regularization
                    

                    loss.backward()
                    # optimizer.step()

                    if((i+1)%accumulation_steps)==0:
                        optimizer.step() # update parameters of net
                        # scheduler.step() # 更新学习率
                        # writer.add_scalar(tag='lr',step=steps,value=scheduler.get_lr()[0])
                        optimizer.zero_grad() # reset gradient
                    if steps % 200 == 0:
                        loss_num = loss.item() * accumulation_steps
                        writer.add_scalar(tag="loss",step=steps,value=loss_num)
                        self.loggger.info("steps:{}/{}, epochs: {}/{}, loss: {}".format(
                            steps,self.num_train_steps,epoch + 1,self.epochs,loss_num
                        ))
                    elif steps % 500 == 0:
                        self.loggger.info("添加直方图，当前steps：{}".format(steps))
                        writer.add_histogram(
                            tag="u's embed",values=VCWE_model.u_embeddings.weight.cpu().data.numpy(),
                            step=steps,buckets=10
                        )
                        writer.add_histogram(
                            tag="v's embed",values=VCWE_model.v_embeddings.weight.cpu().data.numpy(),
                            step=steps,buckets=10
                        )

                    steps += 1
                    epoch_steps += 1

                    # sys.exit(1)

                    # if i > 0 and i % 1000 == 0:
                    #     print('loss=', running_loss/1000)
                    #     running_loss=0.0
                # print("要结束了哦")
                # sys.exit(0)
            # 在这里进行评估rho
            rho,rho_list = self.rho_acc(VCWE_model.u_embeddings.weight.cpu().data.numpy(),self.data.id2word)

            self.loggger.info("epoch: {}, avg_epoch_loss: {}".format(epoch+1,running_loss/epoch_steps))
            if (epoch+1) % 5 == 0 or (epoch+1) == self.epochs:
                VCWE_model.save_embedding(self.data.id2word, emb_dir+"zh_wiki_VCWE_ep"+str(epoch+1)+".txt")
                # state = {
                #     'model':VCWE_model.state_dict(),
                #     'optimizer':optimizer.state_dict(),
                #     'epoch': epoch
                # }
                # torch.save(state,"./model/{}_{}_vcwe_parame".format(self.exp_name,epoch+1))

            # # TODO 保存检查点
            # with tune.checkpoint_dir(epoch) as checkpoint_dir:
            #     path = os.path.join(checkpoint_dir,"checkpoint")
            #     torch.save((VCWE_model.state_dict(),optimizer),path)
            # TODO 报告实验记录
            tune.report(
                rho=rho,loss=running_loss/epoch_steps,
                ch240=rho_list[0],ch297=rho_list[1],mc30=rho_list[2],
                rg65=rho_list[3]
            )
            


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file",
                        default="./data/zh_wiki.txt",
                        type=str,
                        required=True,
                        help="The input file that the VCWE model was trained on.")    
    parser.add_argument("--vocab_file",
                        default="./data/vocabulary.txt",
                        type=str,
                        required=True,
                        help="The vocabulary file that the VCWE model was trained on.")
    parser.add_argument("--img_data_file",
                        default="./data/char_img_sub_mean.npy",
                        type=str,
                        help="The image data file that the VCWE model was trained on.") 
    parser.add_argument("--char2ix_file",
                        default="./data/char2ix.npz",
                        type=str,
                        help="The character-to-index file corespond to the image data file.")                         
    parser.add_argument("--output_dir",
                        default="./embedding/",
                        type=str,
                        help="The output directory where the embedding file will be written.")   
    parser.add_argument("--line_batch_size",
                        default=32,
                        type=int,
                        help="Batch size for lines.")
    parser.add_argument("--sample_batch_size",
                        default=128,
                        type=int,
                        help="Batch size for samples in a line.")      
    parser.add_argument("--emb_dim",
                        default=100,
                        type=int,
                        help="Embedding dimensions.")     
    parser.add_argument("--maxwordlength",
                        default=5,
                        type=int,
                        help="The maximum number of characters in a word.")  
    parser.add_argument("--neg_num",
                        default=5,
                        type=int,
                        help="The number of negative samplings.") 
    parser.add_argument("--window_size",
                        default=5,
                        type=int,
                        help="The window size.") 
    parser.add_argument("--discard",
                        default=1e-5,
                        type=int,
                        help="The sub-sampling threshold.") 
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', 
                        type=int, 
                        default=12345,
                        help="random seed for initialization")
    parser.add_argument('--exp_name', 
                        type=str, 
                        default="临港大道",
                        help="实验名称")                            
    args = parser.parse_args()
    w2v = Word2VecTrainer(input_file = args.input_file, \
                          vocabulary_file = args.vocab_file, \
                          img_data_file = args.img_data_file, \
                          char2ix_file = args.char2ix_file, \
                          output_dir = args.output_dir, 
                          maxwordlength = args.maxwordlength,
                          emb_dimension = args.emb_dim, 
                          line_batch_size = args.line_batch_size,
                          sample_batch_size = args.sample_batch_size,
                          neg_num = args.neg_num,
                          window_size = args.window_size,
                          discard = args.discard,
                          epochs = args.num_train_epochs,
                          initial_lr = args.learning_rate,
                          seed = args.seed,
                          exp_name=args.exp_name)
    # print("---?实验名称",args.exp_name)
    # sys.exit(0)
    # w2v.train()
    w2v.train_tune(num_samples=4)
    
if __name__ == "__main__":
    main()
                
