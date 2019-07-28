import numpy as np
import tensorflow as tf
from itertools import permutations

N = 16
K = 8
Rc = K/N
Rm = 1 # BPSK
ebn0 = np.arange(0,6,1,dtype=np.float32)
ebn0_num = 10**(ebn0/10)
SNR = ebn0 + 10*np.log10(Rc*Rm) + 10*np.log10(2)
SNR_num = 10**(SNR/10)

word_seed = 786000
noise_seed = 345000
wordRandom = np.random.RandomState(word_seed)
noiseRandom = np.random.RandomState(noise_seed)
validation_ratio = 0.2
numOfWord = 600
# numOfWord = 1
batch_size = numOfWord*len(ebn0)
batches_train = int(np.round(125/len(ebn0))*len(ebn0))
batches_test = int(np.round(300/len(ebn0))*len(ebn0))
batches_val = int(batches_train*validation_ratio)
batches_train = int(batches_train*(1-validation_ratio))
patience = 10

n = int(np.log2(N))
Fi = np.ones([1])
for i in range(n):
    Fi = np.vstack((np.hstack((Fi,Fi*0)),np.hstack((Fi,Fi))))
F_kron_n = Fi
combins = [np.array(c) for c in permutations(np.arange(0,n,1),n)]

indices = np.loadtxt('FrozenBit/'+str(N)+'.txt',dtype=int)-1
FZlookup = np.zeros((N))
FZlookup[indices[:K]] = -1
bitreversedindices = np.zeros((N),dtype=int)
for i in range(N):
    b = '{:0{width}b}'.format(i, width=n)
    bitreversedindices[i] = int(b[::-1], 2)

FER = np.zeros(len(ebn0))
BER = np.zeros(len(ebn0))

def fFunction(a,b):
    c = tf.sign(a)*tf.sign(b)*tf.minimum(tf.abs(a),tf.abs(b))
    return c

def get_weight(sess, LV, RV):
    LWeight = sess.run(LV)
    RWeight = sess.run(RV)
    return LWeight, RWeight

def assign_weight(LWeight, RWeight, sess, LV, RV):
    sess.run(LV.assign(LWeight))
    sess.run(RV.assign(RWeight))
    
def quantize(arr,binary_prec):   
    val = 2**(-1*binary_prec+1)
    arr = np.floor(arr/val)*val
    return arr

def quantizeToClosestBinary(arr,binary_prec):
    val = 2**(-1*binary_prec+1)
    
    arr = np.round(arr/val)*val
    arr[arr<0] = 0
    return arr

def quantizeToBins(arr,bin_bit):
    remove_1 = np.setdiff1d(arr,np.ones((1))) # remove 1
    binvals = [np.percentile(remove_1,pr) for pr in np.linspace(0,100,2**(bin_bit))]
    binvals = binvals
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                    arr[i,j,k] = min(binvals, key=lambda x:abs(x - arr[i,j,k]))
    return arr

def BINARYquantizeToBins(arr,bin_bit,binary_prec):
    BINarr = quantizeToClosestBinary(arr,binary_prec)
    unique, counts = np.unique(BINarr,return_counts=True)
    idx = np.argsort(counts)
#     print('Unique:',unique)
#     print('Counts:',counts)
#     print(idx)
    if(len(unique) > 2**bin_bit):
        binvals = unique[idx[-2**bin_bit:]]
    else:
        binvals = unique
#     print('Binvals:',binvals)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                    arr[i,j,k] = min(binvals, key=lambda x:abs(x - arr[i,j,k]))
    return arr

def gendata(const,flag,perm):
    X = np.zeros((len(ebn0)*numOfWord,N))
    Y = np.zeros((len(ebn0)*numOfWord,K))
    
    for i in range(len(ebn0)):
        u = wordRandom.randint(2,size=(numOfWord*K))
#         u = np.zeros((numOfWord*K))
        x = np.tile(FZlookup.copy(),(numOfWord))
        x[x==-1] = u # -1's will get replaced by message bits below
        x = np.reshape(x,(-1,N))
        if not(perm):
            x = x[:,bitreversedindices] # bit reversal
        x = np.mod(np.matmul(x,F_kron_n),2) # encode
        tx_waveform = 2*x-1 # bpsk
        if(flag):
            rx_waveform = noiseRandom.normal(0.0,1.0,tx_waveform.shape)*np.sqrt(1/SNR_num[const]) + tx_waveform
            initia_llr = -2*rx_waveform*SNR_num[const] #away 0
        else:
            rx_waveform = noiseRandom.normal(0.0,1.0,tx_waveform.shape)*np.sqrt(1/SNR_num[i]) + tx_waveform   
            initia_llr = -2*rx_waveform*SNR_num[i] #away 0

        X[i*numOfWord:(i+1)*numOfWord,:] = initia_llr
        Y[i*numOfWord:(i+1)*numOfWord,:] = np.reshape(u,(-1,K))
    return X, Y