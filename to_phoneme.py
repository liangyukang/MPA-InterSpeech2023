from tqdm import tqdm
from g2p_en import G2p

g2p=G2p()
for filename in ["dev-clean.tsv",'dev-other.tsv',
             'test-clean.tsv','test-other.tsv',
             'train-clean-100.tsv','train-clean-360.tsv','train-other-500.tsv']:
    with open(filename,'r') as f:
        outfile='phoneme/'+filename
        out=open(outfile,'w')

        for i,sample in tqdm(enumerate(f.readlines())):
            if i==0:
                out.write(sample)
            else:
                features=sample.strip().split('\t')
                phoneme=g2p(features[3])
                features[3]=' '.join([p for p in phoneme if p !=' '])

                out.write("%s\n" %('\t'.join(features)))
            
