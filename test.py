import numpy as np
from PIL import Image
import os
import pickle as pkl

def pkl_dump(data,f_path):
    pkl.dump(data,open(f_path,'wb'))
def pkl_load(f_path):
    with open(f_path,'rb') as f:
        return pkl.load(f)
def numpy2image(np_array,name="fuck.jpg"):
    path = "test/fontImage/"

    if not os.path.exists(path):
        os.makedirs(path)
    size = np_array.size[::-1]
    a = np.asarray(np_array).reshape(size)
    
    img = Image.fromarray(np.uint8(a*255))
    # img.show()
    name += ".jpg"
    img.save(os.path.join(path,name))
def test1():
    data = np.load('./data/char2ix.npz',allow_pickle=True)


    print(data.files)
    char_to_ix = data['char_to_ix'].item()
    ix_to_char = data['ix_to_char'].item()

    print(len(char_to_ix))
    print(ix_to_char) # 1-5031，对应5031个字

def test2(img_data_file="./data/char_img_sub_mean.npy"):
    '''
    读取image文件
    '''
    img_data = np.load(img_data_file)
    char2ix_file = "./data/char2ix.npz"
    data = np.load(char2ix_file,allow_pickle=True)
    char2id = data['char_to_ix'].item()
    # print(char2id)
    char = ['淇','镇','药','倾']
    char1 = char2id['淇']

    img_0 = img_data[0]
    print(img_0)
    # print(img_0.sum())

    char1 = img_data[char1]
    # print(char1.sum())

    char11 = img_0 + char1
    print(img_0.sum())
    print(char1.sum())
    print(char11.sum())

    # img = Image.fromarray(char11.squeeze())
    # img.convert("RGB").save("fuck.jpg")
    # id2char = {v:k for k,v in char2id.items()}
    
    # print(id2char[0])
    for ch,ch_id in char2id.items():
        # print(ch_id)
        # if ch_id == 0:
        #     print(ch)/
        ch_image = img_data[ch_id]
        # print(ch_image.squeeze().shape)
        # break
        img = Image.fromarray(ch_image.squeeze()*255)
        img.convert("RGB").save("{}_{}.jpg".format(ch,ch_id))

    # for i in range(len(char)):
    #     # print(char[i])
    #     c_id = char2id[char[i]]
    #     first_image = img_data[c_id]
    #     print("----")
    #     print(first_image.sum())
    #     first_image_squeeze = img_data[c_id].squeeze()

    #     img = Image.fromarray(first_image_squeeze)
    #     img.convert("RGB").save("fuck.jpg")

        # break
        # img = Image.fromarray(first_image)
        # img.convert('L').save(os.path.join("test/fontImage/",'char_%d.jpg' %(i)))

    # for i in range(100,200):

    #     first_image = img_data[0].squeeze()
    #     print(first_image.shape)
    #     img = Image.fromarray(first_image)
    #     # im.show()
    #     img.convert('L').save(os.path.join("test/fontImage/",'fuck_%d.jpg' %(i)))
        # break
        # numpy2image(first_image,"fuck %d.jpg" % (i))

    # print(first_image.sum())

if __name__ == "__main__":
    test2()
