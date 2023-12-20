from tqdm import tqdm
from g2p_en import G2p
import json

g2p=G2p()
lenOfPhotokens={} #记录所有文件photoken长度
dictg2p={}
dictp2g={}
for filename in ["dev-clean.tsv",'dev-other.tsv',
             'test-clean.tsv','test-other.tsv']:
             #'train-clean-100.tsv','train-clean-360.tsv','train-other-500.tsv']:
    with open(filename,'r') as f:
        outfile='phoneme_map/'+filename
        out=open(outfile,'w')
        lenOfPhotokens_sample=[] #记录一个文件中每一个样本的photoken长度
        for i,sample in tqdm(enumerate(f.readlines())):
            if i==0:
                out.write(sample)
            else:
                features=sample.strip().split('\t')
                tokens=features[3].split(' ')
                
                phoneme=g2p(features[3])

                lens=[]    #记录每个样本中的phoneme token的长度
                photokens=[] #记录每个样本中的phoneme token
                
                tmppho=[]
                length=0
                
                for p in phoneme:
                    if p==' ':
                        lens.append(length)
                        photokens.append(" ".join(tmppho))
                        length=0
                        tmppho=[]
                    else:
                        tmppho.append(p)
                        length+=1
                        
                lens.append(length)
                photokens.append(" ".join(tmppho))

                for token,photoken in zip(tokens,photokens):
                    dictg2p[token]=photoken
                    dictp2g[photoken]=token
                
                lenOfPhotokens_sample.append(lens)
                features[3]=' '.join([p for p in phoneme if p !=' '])

                out.write("%s\n" %('\t'.join(features)))

        lenOfPhotokens[filename]=lenOfPhotokens_sample

dictg2pfile=open('phoneme_map/dictg2p.txt','w')
dictp2gfile=open('phoneme_map/dictp2g.txt','w')
lenfile=open('phoneme_map/len.txt','w')

dictg2pfile.write(json.dumps(dictg2p))
dictp2gfile.write(json.dumps(dictp2g))
lenfile.write(json.dumps(lenOfPhotokens))
