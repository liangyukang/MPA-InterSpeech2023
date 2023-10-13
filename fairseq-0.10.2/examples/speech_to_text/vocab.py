import os.path as op
from tqdm import tqdm
path='/home/v-yukliang/mount_2/v-yukliang/yukangliang/DataSets/WeNetBlob/smalldataset'
tokens=set()
out=open(op.join(path,'dict.txt'),'w')
for split in ['train','val','test']:
    filename=op.join(path,split+'.tsv')
    for line in tqdm(open(filename,'r').readlines()[1:]):
        _,_,_,tgt_text,_ = line.strip().split('\t')
        text=tgt_text.split(' ')
        # print(text)
        tokens.update(text)
for token in tokens:
    out.write("%s 1\n" %(token))
