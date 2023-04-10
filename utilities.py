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
import IPython.display as ipd
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

def load_train(nb_per_class,duration=2,maxfreq=5000):
    """
    Charge les données.
    Paramètres :
        - nb_per_class : nombres de fichiers audios utilisés pour chaque classe d'instrument
        - duration : duree pris en compte pour un fichier audio
        - maxfreq : nombre de tranche de frequences utilisee pour la decomposition de Fourier
    Return:
        DataFrame dont les lignes sont les fichiers audios, les colonnes le label et l'intensite pour chaque frequence

    """
    
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

def load_subsets(df,coef_train=0.6, coef_test=0.4):
    """
    Fait la séparation des données entre un jeu pour les entrainements
        (cross-validation comprise) et tests.
    Return : X_train, y_train, X_valid, y_valid
    """
    guitar = df[df['Class'] == "Sound_Guitar"].sample(frac=1)
    drum = df[df['Class'] == "Sound_Drum"].sample(frac=1)
    violin = df[df['Class'] == "Sound_Violin"].sample(frac=1)
    piano = df[df['Class'] == "Sound_Piano"].sample(frac=1)
    
    column_headers = df.columns.values[2:]
    Ntot   = len(guitar)
    Ntrain = int(coef_train*Ntot)
    Ntest = int(coef_test*Ntot)
    data_train = pd.DataFrame()
    data_valid = pd.DataFrame()
    for i in tqdm(range(Ntrain)):
        # je n ai pas trouve de moyen moins CHIANT de faire ca
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(drum.iloc[i]))])
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(guitar.iloc[i]))])
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(piano.iloc[i]))])
        data_train = pd.concat([data_train, pd.DataFrame.transpose(pd.DataFrame(violin.iloc[i]))])

    for i in tqdm(range(Ntrain, Ntrain + Ntest)):
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
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    ax.plot(CumulativeExplainedVariance, marker='+')
    ax.set_ylabel("Variance")
    ax.set_xlabel("n_component")
        
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
    
    
def cross_val(X,y,cv,C,degree,begin,end,step):
    clf = sklearn.svm.SVC(C=C,kernel="poly",degree=degree,coef0=1,probability=True)
    train_score = []
    valid_score = []
    nComp_range = np.arange(begin,end,step)
    for i in tqdm(nComp_range):
        preProc = sklearn.decomposition.PCA(i)
        preProc.fit(X)

        X_Transformed = preProc.transform(X)

        cv_results = cross_validate(clf, X_Transformed, y, cv=10,return_estimator=True,return_train_score=True)
        #sorted(cv_results.keys())
        train_score.append(np.max(cv_results['train_score']))
        valid_score.append(np.max(cv_results['test_score']))
    bestIndex = np.argmax(valid_score)
    bestComp = nComp_range[bestIndex]
    
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(nComp_range, train_score, label= "train score")
    ax.plot(nComp_range, valid_score, label= "valid score")
    ax.set_xlabel("nombre comp")
    ax.set_ylabel("scores")
    ax.legend()
    ax.set_ylim([0.5,1])
    
    
    preProc = sklearn.decomposition.PCA(bestComp)
    preProc.fit(X)
    X_Transformed = preProc.transform(X)

    cv_results = cross_validate(clf, X_Transformed, y, cv=10,return_estimator=True,return_train_score=True)
    #sorted(cv_results.keys())
    best_estim_index = np.argmax(cv_results["test_score"])
    
    print(f"best n_comp : {nComp_range[bestIndex]} | train score : {train_score[bestIndex]} | valid score : {valid_score[bestIndex]}")
    
    return cv_results["estimator"][best_estim_index], preProc

def find_real(i,y_test):
    if  y_test.iloc[i] == "Sound_Drum":
        return 0
    elif  y_test.iloc[i] == "Sound_Guitar":
        return 1
    elif  y_test.iloc[i] == "Sound_Piano":
        return 2
    else:
        return 3
    
def show_bad_prediction(X_test_Transformed,y_test,best_svc,data,X_test,show_file_name=False):
    predict = best_svc.predict(X_test_Transformed)
    predict_proba = best_svc.predict_proba(X_test_Transformed)
    for i in range(len(predict)):
        if {predict[i]} != {y_test.iloc[i]}:
            proba = np.max(predict_proba[i])
            prob_real = predict_proba[i][find_real(i,y_test)]
            if show_file_name:
                name_file = data.loc[int(X_test.iloc[i].name)]["FileName"]
                print(f"file : {name_file:20s}   |   predict : {predict[i]:12s} with prob = {proba:.3f}   |   real : {y_test.iloc[i]:12s} with prob = {prob_real:.3f}")
            else:
                print(f"predict : {predict[i]:12s} with prob = {proba:.3f}   |   real : {y_test.iloc[i]:12s} with prob = {prob_real:.3f}")


def class_file(file,duration):
    dic = {}
    l = 0
    flag = True
    dur = librosa.get_duration(filename=file)
    if dur > 2*duration:
        mid = dur/2.
    else:
        mid = 0
    signal, rate = librosa.load(file,offset=mid,duration=duration)
        #newS, newR = extract2s(signal, rate)
    fft1,fft2 = calc_fft(signal,rate, maxFreq=5000)
    fft1 = cut_in_parts(fft1,10)
    fft2 = cut_in_parts(fft2,10)
    if(flag):
        l = len(fft1)
        flag = False
        for j in range(l):
            dic[str(j/2.)] = [] # a changer si on choisit d autre frequences
    for i in range(l):
        dic[str(i/2.)].append(fft2[i]) # a changer si on choisit d autre frequences
    return pd.DataFrame(dic)

def predict_instrument(file,best_svc,preProc,prob=False):
    audio = class_file(file,2)
    if prob:
            p = np.max(best_svc.predict_proba(preProc.transform(audio))[0])*100
            print(f"It's a {best_svc.predict(preProc.transform(audio))[0]} ! (with {p:.1f}% accuracy)")
    else:
        print(f"It's a {best_svc.predict(preProc.transform(audio))[0]} !")