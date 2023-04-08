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
import sklearn
from tqdm.notebook import tqdm, trange
import time 
import ipywidgets as widgets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_validate

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

def cut_in_parts(tab, n):
    return [tab[n*i] for i in range(0, int(len(tab)/n))]

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
    for file_name in tqdm(df_exp["FileName"]):
        # changement L-A
        dur = librosa.get_duration(filename="./dataset/Train_submission/Train_submission/"+file_name)
        if dur > 2*duration:
            mid = dur/2.
        else:
            mid = 0
        signal, rate = librosa.load("./dataset/Train_submission/Train_submission/"+file_name,offset=mid,duration=duration)
        #newS, newR = extract2s(signal, rate)
        fft1,fft2 = calc_fft(signal,rate, maxFreq=maxfreq)
        fft1 = cut_in_parts(fft1,10)
        fft2 = cut_in_parts(fft2,10)
        if(flag):
            l = len(fft1)
            flag = False
            for j in range(l):
                dic[str(j/2.)] = [] # a changer si on choisit d autre frequences
        for i in range(l):
            dic[str(i/2.)].append(fft2[i]) # a changer si on choisit d autre frequences
    df_prim = pd.DataFrame(dic, index=df_exp.index[0:4*nb_per_class])
    df_exp = pd.concat((df_exp, df_prim), axis=1)
    return df_exp

def load_test(nb_per_class,begin=-1,duration=2,maxfreq=5000):
    
    if nb_per_class > 20:
        raise ValueError("too many files")
    df_test = pd.read_csv("./dataset/Metadata_Test.csv")
    guitar = df_test[df_test['Class'] == "Sound_Guiatr"]
    drum = df_test[df_test['Class'] == "Sound_Drum"]
    violin = df_test[df_test['Class'] == "Sound_Violin"]
    piano = df_test[df_test['Class'] == "Sound_Piano"]
    df_exp = pd.concat([violin[0:nb_per_class], drum[0:nb_per_class], piano[0:nb_per_class], guitar[0:nb_per_class]])
    frequencies = None
    df_prim = pd.DataFrame
    dic = {}
    l = 0
    flag = True
    for file_name in tqdm(df_exp["FileName"]):
        #si begin = -1, on tape au milieu du fichier (presupose de tout charger)
        # changement L-A
        dur = librosa.get_duration(filename="./dataset/Test_submission/Test_submission/"+file_name)
        if dur > 2*duration:
            mid = dur/2.
        else:
            mid = 0
        signal, rate = librosa.load("./dataset/Test_submission/Test_submission/"+file_name,offset=mid,duration=duration)
        #newS, newR = extract2s(signal, rate)
        fft1,fft2 = calc_fft(signal,rate, maxFreq=maxfreq)
        fft1 = cut_in_parts(fft1,10)
        fft2 = cut_in_parts(fft2,10)
        if(flag):
            l = len(fft1)
            flag = False
            for j in range(l):
                dic[str(j/2.)] = [] # a changer si on choisit d autre frequences
        for i in range(l):
            dic[str(i/2.)].append(fft2[i]) # a changer si on choisit d autre frequences
    df_prim = pd.DataFrame(dic, index=df_exp.index[0:4*nb_per_class])
    df_exp = pd.concat((df_exp, df_prim), axis=1)
    return df_exp

def convert_df(df):
    df = df.sample(frac=1) # melange tout
    column_headers = df.columns.values[2:]    
    X = df[column_headers]
    y = df["Class"]
    
    return X, y

def load_subsets(df,coef_train=0.6, coef_valid=0.4):
    guitar = df[df['Class'] == "Sound_Guitar"]
    drum = df[df['Class'] == "Sound_Drum"]
    violin = df[df['Class'] == "Sound_Violin"]
    piano = df[df['Class'] == "Sound_Piano"]

    column_headers = df.columns.values[2:]
    coef_test = 1 - coef_train - coef_valid
    Ntot   = len(guitar)
    Ntrain = int(coef_train*Ntot)
    Nvalid = int(coef_valid*Ntot)
    Ntest  = Ntot - Ntrain - Nvalid
    data_train = pd.DataFrame()
    data_valid = pd.DataFrame()
    for i in tqdm(range(Ntrain)):
        # je n ai pas trouve de moyen moins CHIANT de faire ca
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(drum.iloc[i]))])
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(guitar.iloc[i]))])
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(piano.iloc[i]))])
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(violin.iloc[i]))])

    for i in tqdm(range(Ntrain, Ntrain + Nvalid)):
        data_valid = pd.concat([data_valid, pd.DataFrame.transpose(pd.DataFrame(drum.iloc[i]))])
        data_valid = pd.concat([data_valid, pd.DataFrame.transpose(pd.DataFrame(guitar.iloc[i]))])
        data_valid = pd.concat([data_valid, pd.DataFrame.transpose(pd.DataFrame(piano.iloc[i]))])
        data_valid = pd.concat([data_valid, pd.DataFrame.transpose(pd.DataFrame(violin.iloc[i]))])

    column_headers = df.columns.values[2:]
    X_train = data_train[column_headers]
    y_train = data_train["Class"]
    X_valid = data_valid[column_headers]
    y_valid = data_valid["Class"]
    return X_train, y_train, X_valid, y_valid

def load_subsets_test(df):

    column_headers = df.columns.values[2:]    
    X_test = df[column_headers]
    y_test = df["Class"]
    
    return X_test, y_test

def plot_cumexpvar(X_train,varianceExplained=0.95):
    preProc = sklearn.decomposition.PCA(varianceExplained)
    preProc.fit(X_train)
    CumulativeExplainedVariance = np.cumsum(preProc.explained_variance_ratio_)
    plt.plot(CumulativeExplainedVariance, marker='+')
    plt.xlabel("Variance")
    plt.ylabel("n_component")
        
def plot_score_components(X_train,y_train,X_valid,y_valid,C,degree,begin,end,step):
    
    assert (begin >= 10)
    linear_training_score = []
    linear_valid_score = []
    clf = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(C=C,kernel="poly",degree=degree,coef0=1))
    #clf = sklearn.svm.SVC(C=C,kernel="poly",degree=degree,coef0=1)
    nComp_range = np.arange(begin,end,step)

    for nC in tqdm(nComp_range):
        preProc = sklearn.decomposition.PCA(nC)
        preProc.fit(X_train)

        X_train_Transformed = preProc.transform(X_train)
        X_valid_Transformed = preProc.transform(X_valid)

        clf.fit(X_train_Transformed, y_train)
        trainscore = clf.score(X_train_Transformed,y_train)
        validscore = clf.score(X_valid_Transformed,y_valid)
        linear_training_score.append(trainscore)
        linear_valid_score.append(validscore)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(nComp_range, linear_training_score, label= "train score")
    ax.plot(nComp_range, linear_valid_score, label= "valid score")
    ax.set_xlabel("nombre comp")
    ax.set_ylabel("scores")
    ax.legend()
    ax.set_ylim([0.5,1])
    bestIndex = np.argmax(linear_valid_score) 
    print(f"best n_comp : {nComp_range[bestIndex]} | train score : {linear_training_score[bestIndex]} | valid score : {linear_valid_score[bestIndex]}")


