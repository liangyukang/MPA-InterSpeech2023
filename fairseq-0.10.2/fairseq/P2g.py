import json

def p2g(datapath,input,filename,id):
    dictp2gfile=datapath+'/dictp2g.txt'
    lenfile=datapath+'/len.txt'

    dictp2g=json.loads(open(dictp2gfile,'r').readlines()[0])
    len=json.loads(open(lenfile,'r').readlines()[0])

    phoneme=input.strip().split(' ')
    filename=filename+'.tsv'

    lengths=len[filename][id]
    photokens=[]
    for l in lengths:
        pho = phoneme[:l]
        phoneme=phoneme[l:]
        photokens.append(' '.join(pho))
    
    # for p in photokens:
    #     p in dictp2g:

    tokens=[dictp2g[p] if p in dictp2g else 'UNK' for p in photokens]
 
    output=' '.join(tokens)
    return output