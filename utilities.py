import glob
import io
import os
from scipy.io import wavfile
import numpy as np
import IPython
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
import librosa


def calc_fft(y,sr, maxFreq=-1):
    """
    Calcul la transformee de fourier de y qui est sample a rate sr
    L analyse est tronquee a maxFreq
    """
    n=len(y)
    freq=np.fft.rfftfreq(n,d=1/sr)
    Y=abs(np.fft.rfft(y)/n)
    if maxFreq==-1:
        return (freq,Y)
    else:
        return (freq[0:maxFreq], Y[0:maxFreq])
    
def extract2s(signal, rate):
    lengh = len(signal)
    if lengh <= 2*rate:
        return signal, rate
    else:
        return signal[int(lengh/2) - rate:int(lengh/2) + rate], rate
    

def load_train(nb_per_class,begin=0,duration=2,maxfreq=5000):
    
    if nb_per_class > 700:
        raise ValueError("too many files")
    df_train = pd.read_csv("./dataset/Metadata_Train.csv")
    guitar = df_train[df_train['Class'] == "Sound_Guitar"]
    drum = df_train[df_train['Class'] == "Sound_Drum"]
    violin = df_train[df_train['Class'] == "Sound_Violin"]
    piano = df_train[df_train['Class'] == "Sound_Piano"]
    df_exp = pd.concat([violin[0:nb_per_class], drum[0:nb_per_class], piano[0:nb_per_class], guitar[0:nb_per_class]])
    frequencies = None
    df_prim = pd.DataFrame
    dic = {}
    l = 0
    flag = True
    for file_name in df_exp["FileName"]:
        signal, rate = librosa.load("./dataset/Train_submission/Train_submission/"+file_name,offset=begin,duration=duration)
        #newS, newR = extract2s(signal, rate)
        fft1,fft2 = calc_fft(signal,rate, maxFreq=maxfreq)
        if(flag):
            l = len(fft1)
            flag = False
            for j in range(l):
                dic[str(j/2.)] = [] # a changer si on choisit d autre frequences
        for i in range(l):
            dic[str(i/2.)].append(fft2[i]) # a changer si on choisit d autre frequences
    print(dic)
    df_prim = pd.DataFrame(dic, index=df_exp.index[0:4*nb_per_class])
    df_exp = pd.concat((df_exp, df_prim), axis=1)
    return df_exp

def load_subsets(df,coef_train=0.6, coef_valid=0.2):
    guitar = df[df['Class'] == "Sound_Guitar"]
    drum = df[df['Class'] == "Sound_Drum"]
    violin = df[df['Class'] == "Sound_Violin"]
    piano = df[df['Class'] == "Sound_Piano"]

    column_headers = df.columns.values[2:]
    coef_test = 1 - coef_train - coef_valid
    assert(coef_test>0) ## on vérifie qu'il reste des exemples pour le test set
    Ntot   = len(guitar)
    Ntrain = int(coef_train*Ntot)
    Nvalid = int(coef_valid*Ntot)
    Ntest  = Ntot - Ntrain - Nvalid
    data_train = pd.DataFrame()
    data_valid = pd.DataFrame()
    data_test = pd.DataFrame()
    for i in range(Ntrain):
        # je n ai pas trouve de moyen moins CHIANT de faire ca
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(drum.iloc[i]))])
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(guitar.iloc[i]))])
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(piano.iloc[i]))])
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(violin.iloc[i]))])

    for i in range(Ntrain, Ntrain + Nvalid):
        data_valid = pd.concat([data_valid, pd.DataFrame.transpose(pd.DataFrame(drum.iloc[i]))])
        data_valid = pd.concat([data_valid, pd.DataFrame.transpose(pd.DataFrame(guitar.iloc[i]))])
        data_valid = pd.concat([data_valid, pd.DataFrame.transpose(pd.DataFrame(piano.iloc[i]))])
        data_valid = pd.concat([data_valid, pd.DataFrame.transpose(pd.DataFrame(violin.iloc[i]))])

    for i in range(Ntrain + Nvalid, Ntot):
        data_test = pd.concat([data_test, pd.DataFrame.transpose(pd.DataFrame(drum.iloc[i]))])
        data_test = pd.concat([data_test, pd.DataFrame.transpose(pd.DataFrame(guitar.iloc[i]))])
        data_test = pd.concat([data_test, pd.DataFrame.transpose(pd.DataFrame(piano.iloc[i]))])
        data_test = pd.concat([data_test, pd.DataFrame.transpose(pd.DataFrame(violin.iloc[i]))])

    column_headers = df.columns.values[2:]
    X_train = data_train[column_headers]
    y_train = data_train["Class"]
    X_test = data_test[column_headers]
    y_test = data_test["Class"]
    X_valid = data_valid[column_headers]
    y_valid = data_valid["Class"]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


        



