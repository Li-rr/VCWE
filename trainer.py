import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm
import numpy as np
import argparse
import sys
from PIL import Image
from scipy import misc
from torchsummary import summary
from visualdl import LogWriter # 百度可视化工具
import datetime
from torch.optim import lr_scheduler

from utils import id2word,pkl_load
from data_reader import DataReader, Word2vecDataset
from model import VCWEModel
from optimization import VCWEAdam
from logginger import init_logger 
import random

class Word2VecTrainer:
    def __init__(self, input_file, vocabulary_file, img_data_file, char2ix_file, output_dir, maxwordlength, emb_dimension, line_batch_size, sample_batch_size, 
                neg_num, window_size, discard, epochs, initial_lr, seed,exp_name):
                 
        torch.manual_seed(seed)
        random.seed(seed)
        # self.img_data = np.load(img_data_file)
        self.img_data = pkl_load(img_data_file)
        self.data = DataReader(input_file, vocabulary_file, char2ix_file, maxwordlength, discard, seed)
        # sample_batch_size，用于构建基本的数据集
        dataset = Word2vecDataset(self.data, window_size, sample_batch_size, neg_num)
        # line_batch_size是一次取几行数据，可以设为1
        self.dataloader = DataLoader(dataset, batch_size=line_batch_size,
                                     shuffle=True, num_workers=0, collate_fn=dataset.collate)

        self.output_dir = output_dir
        self.emb_size = len(self.data.word2id)
        self.char_size = len(self.data.char2id)+1       #5031
        self.emb_dimension = emb_dimension
        self.line_batch_size = line_batch_size # 在DataLoader中使用
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.exp_name = exp_name
        self.loggger = init_logger("myVcwe_{}".format(self.exp_name),"./logs")
        self.VCWE_model = VCWEModel(
            self.emb_size, 
            self.emb_dimension, 
            self.data.wordid2charid, 
            self.char_size,
            self.data.noise_dist,
            self.loggger,
            self.exp_name
        )


        
        
        print("#---------------------------------------")
        print(self.VCWE_model)
        self.use_cuda = True #torch.cuda.is_available()
        
        
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_train_steps= int(len(self.dataloader) * self.epochs)
        if self.use_cuda:
            self.VCWE_model.cuda()


    def train(self):
        # img ([5031, 1, 40, 40]) 第一个维度代表不同的字
        self.img_data = torch.from_numpy(self.img_data).to(self.device)

        if self.img_data.dim() == 3:
            print(self.img_data.shape)
            self.img_data = self.img_data.unsqueeze(1)
            print(self.img_data.shape)
        # sys.exit(0)
        # print(self.img_data[0][0].shape)
        # print("----")

        # print(self.img_data[5].sum())
        # temp2 = self.img_data[0][0].cpu().numpy() * 255
        # # misc.imsave('out.jpg', temp2)        
        # im = Image.fromarray(temp2)
        
        # im = im.convert('L') 

        # im.save('outfile2.png')
        # print(self.img_data.shape)
        cur_time = datetime.datetime.now()
        cur_time = datetime.datetime.strftime(cur_time,'%Y-%m-%d_%H:%M:%S')
        logs_path = "./logs/" + self.exp_name +cur_time
        writer = LogWriter(logdir=logs_path)

        no_decay = ['bias']
        optimizer_parameters = [
             {'params': [p for n, p in self.VCWE_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
             {'params': [p for n, p in self.VCWE_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
             ]
		
        print("num_train_steps=",self.num_train_steps)
        optimizer = VCWEAdam(optimizer_parameters,
                             lr=self.initial_lr,
                             warmup=0.1,
                             t_total=self.num_train_steps)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=1e-4, last_epoch=-1)
        steps = 0
        for epoch in range(self.epochs):

            self.loggger.info("Epoch: " + str(epoch + 1))
       

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
                    optimizer.zero_grad()
                    loss = self.VCWE_model.forward(pos_u, pos_v, neg_v, self.img_data)
                    # print("loss's type",type(loss),loss)
                    running_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                    if steps % 50 == 0:
                        loss_num = loss.item()
                        writer.add_scalar(tag="loss",step=steps,value=loss_num)
                        self.loggger.info("steps:{}/{}, epochs: {}/{}, loss: {}".format(
                            steps,self.num_train_steps,epoch + 1,self.epochs,loss_num
                        ))
                    elif steps % 500 == 0:
                        self.loggger.info("添加直方图，当前steps：{}".format(steps))
                        writer.add_histogram(
                            tag="u's embed",values=self.VCWE_model.u_embeddings.cpu().data.numpy(),
                            step=steps,buckets=10
                        )
                        writer.add_histogram(
                            tag="v's embed",values=self.VCWE_model.v_embeddings.cpu().data.numpy(),
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
            # self.loggger.info("epoch: {}, avg_epoch_loss: {} lr: {}".format(epoch+1,running_loss/epoch_steps,scheduler.get_last_lr()))
            self.loggger.info("epoch: {}, avg_epoch_loss: {}".format(epoch+1,running_loss/epoch_steps))
            if (epoch+1) % 5 == 0 or (epoch+1) == self.epochs:
                output_dir = "/data/home/scv2877/archive/"
                state = {
                    'model':self.VCWE_model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(state,output_dir+"model/{}_{}_vcwe_parame".format(self.exp_name,epoch+1))
                self.VCWE_model.save_embedding(self.data.id2word, output_dir+"zh_wiki_VCWE_ep"+str(epoch+1)+".txt")


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
    w2v.train()
    
if __name__ == "__main__":
    main()
                
