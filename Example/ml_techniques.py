import numpy as np
import pickle as pk
from sklearn import svm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn import metrics
from scipy.stats import spearmanr,kendalltau
import os

def train_test_split(samples,train_prop):
    numsamps = len(samples)
    num_test = int(numsamps*(1-train_prop))
    keys = list(samples.keys())
    test = list(np.random.choice(keys,num_test,replace = False))
    train = [ky for ky in keys if ky not in test]
    return train,test
    
def prep_data_continuous(samples,train_prop = 0.7):
    train,test = train_test_split(samples,train_prop)
    train_df = pd.DataFrame.from_dict(dict([(s,samples[s][1]) for s in train]))
    test_df = pd.DataFrame.from_dict(dict([(s,samples[s][1]) for s in test]))
    test_df = test_df.loc[train_df.index]
    train_target = np.array([samples[s][0] for s in train])
    test_target = np.array([samples[s][0] for s in test])
    return (train_df.T.values,train_target),(test_df.T.values,test_target)

def prep_data_discrete(samples,train_prop = 0.7):
    train,test = train_test_split(samples,train_prop)
    train_df = pd.DataFrame.from_dict(dict([(s,samples[s][1]) for s in train]))
    test_df = pd.DataFrame.from_dict(dict([(s,samples[s][1]) for s in test]))
    test_df = test_df.loc[train_df.index]
    train_target = np.array([samples[s][0] for s in train])
    test_target = np.array([samples[s][0] for s in test])
    return ((train_df.T.values>10**-6).astype(float),train_target),((test_df.T.values>10**-6).astype(float),test_target)

    # def safe_divide(a,b,er = 0):
    # if b:
    #     return a/b
    # else:
    #     return er


if __name__=="__main__":

    N = 1000

    experiments = ["C.Difficile (Cont.)","C.Difficile (Disc.)","B.longum (Cont.)","B.longum (Disc.)"]
    cols = ["SVM Miss","SVM TPR","SVM FPR","SVM AUROC","RF Miss","RF TPR","RF FPR","RF AUROC"]
    ml_techniques_binary_mean = pd.DataFrame(index = experiments, columns = cols)
    ml_techniques_binary_var = pd.DataFrame(index = experiments, columns = cols)

    ml_Lplantarum = pd.DataFrame(index = ["Continuous","Discrete"], columns = ["SVM Avg Error","SVM Variance Error","SVM Kendall","SVM Spearman","RF Avg Error","RF Kendall","RF Spearman"])


    with open("friendlySamples/upto_species/cdiff_species_dict.pk","rb") as fl:
        samples = pk.load(fl)

    tot_miss = np.empty(N)
    true_pos = np.empty(N)
    false_pos = np.empty(N)
    tpok = np.empty(N,dtype = bool)
    fpok = np.empty(N,dtype= bool)
    aucrocs = np.empty(N)
    arok = np.zeros(N).astype(bool)
    for i in range(N):
        train_tp,test_tp = prep_data_continuous(samples)
        classifier = svm.SVC(probability = True)
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum(prediction != test_tp[1])/len(test_tp[1])
        tot_miss[i] = miss_prop
        if sum(test_tp[1] == 1):
            true_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==1) for j in range(len(prediction))])/sum(test_tp[1] == 1)
            tpok[i] = True
        else:
            tpok[i] = False
        if sum(test_tp[1] == 0):
            false_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==0) for j in range(len(prediction))])/sum(test_tp[1] == 0)
            fpok[i] = True
        else:
            fpok[i] = False
        y_score = classifier.predict_proba(test_tp[0])[:,1]
        try:
            aucrocs[i] = metrics.roc_auc_score(test_tp[1],y_score)
            arok[i]=True
        except:
            pass


    ml_techniques_binary_mean.loc["C.Difficile (Cont.)","SVM Miss"] = np.mean(tot_miss)
    ml_techniques_binary_mean.loc["C.Difficile (Cont.)","SVM TPR"] = np.mean(true_pos[tpok])
    ml_techniques_binary_mean.loc["C.Difficile (Cont.)","SVM FPR"] = np.mean(false_pos[fpok])
    ml_techniques_binary_mean.loc["C.Difficile (Cont.)","SVM AUROC"] = np.mean(aucrocs[arok])

    ml_techniques_binary_var.loc["C.Difficile (Cont.)","SVM Miss"] = np.var(tot_miss)
    ml_techniques_binary_var.loc["C.Difficile (Cont.)","SVM TPR"] = np.var(true_pos[tpok])
    ml_techniques_binary_var.loc["C.Difficile (Cont.)","SVM FPR"] = np.var(false_pos[fpok])
    ml_techniques_binary_var.loc["C.Difficile (Cont.)","SVM AUROC"] = np.var(aucrocs[arok])


    tot_miss = np.empty(N)
    true_pos = np.empty(N)
    false_pos = np.empty(N)
    tpok = np.empty(N,dtype = bool)
    fpok = np.empty(N,dtype= bool)
    aucrocs = np.empty(N)
    arok = np.zeros(N).astype(bool)
    for i in range(N):
        train_tp,test_tp = prep_data_continuous(samples)
        classifier = RandomForestClassifier()
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum(prediction != test_tp[1])/len(test_tp[1])
        tot_miss[i] = miss_prop
        if sum(test_tp[1] == 1):
            true_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==1) for j in range(len(prediction))])/sum(test_tp[1] == 1)
            tpok[i] = True
        else:
            tpok[i] = False
        if sum(test_tp[1] == 0):
            false_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==0) for j in range(len(prediction))])/sum(test_tp[1] == 0)
            fpok[i] = True
        else:
            fpok[i] = False
        y_score = classifier.predict_proba(test_tp[0])[:,1]
        try:
            aucrocs[i] = metrics.roc_auc_score(test_tp[1],y_score)
            arok[i]=True
        except:
            pass

   
    ml_techniques_binary_mean.loc["C.Difficile (Cont.)","RF Miss"] = np.mean(tot_miss)
    ml_techniques_binary_mean.loc["C.Difficile (Cont.)","RF TPR"] = np.mean(true_pos[tpok])
    ml_techniques_binary_mean.loc["C.Difficile (Cont.)","RF FPR"] = np.mean(false_pos[fpok])
    ml_techniques_binary_mean.loc["C.Difficile (Cont.)","RF AUROC"] = np.mean(aucrocs[arok])

    ml_techniques_binary_var.loc["C.Difficile (Cont.)","RF Miss"] = np.var(tot_miss)
    ml_techniques_binary_var.loc["C.Difficile (Cont.)","RF TPR"] = np.var(true_pos[tpok])
    ml_techniques_binary_var.loc["C.Difficile (Cont.)","RF FPR"] = np.var(false_pos[fpok])
    ml_techniques_binary_var.loc["C.Difficile (Cont.)","RF AUROC"] = np.var(aucrocs[arok])

    tot_miss = np.empty(N)
    true_pos = np.empty(N)
    false_pos = np.empty(N)
    tpok = np.empty(N,dtype = bool)
    fpok = np.empty(N,dtype= bool)
    aucrocs = np.empty(N)
    arok = np.zeros(N).astype(bool)
    for i in range(N):
        train_tp,test_tp = prep_data_discrete(samples)
        classifier = svm.SVC(probability = True)
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum(prediction != test_tp[1])/len(test_tp[1])
        tot_miss[i] = miss_prop
        if sum(test_tp[1] == 1):
            true_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==1) for j in range(len(prediction))])/sum(test_tp[1] == 1)
            tpok[i] = True
        else:
            tpok[i] = False
        if sum(test_tp[1] == 0):
            false_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==0) for j in range(len(prediction))])/sum(test_tp[1] == 0)
            fpok[i] = True
        else:
            fpok[i] = False
        y_score = classifier.predict_proba(test_tp[0])[:,1]
        try:
            aucrocs[i] = metrics.roc_auc_score(test_tp[1],y_score)
            arok[i]=True
        except:
            pass

    ml_techniques_binary_mean.loc["C.Difficile (Disc.)","SVM Miss"] = np.mean(tot_miss)
    ml_techniques_binary_mean.loc["C.Difficile (Disc.)","SVM TPR"] = np.mean(true_pos[tpok])
    ml_techniques_binary_mean.loc["C.Difficile (Disc.)","SVM FPR"] = np.mean(false_pos[fpok])
    ml_techniques_binary_mean.loc["C.Difficile (Disc.)","SVM AUROC"] = np.mean(aucrocs[arok])

    ml_techniques_binary_var.loc["C.Difficile (Disc.)","SVM Miss"] = np.var(tot_miss)
    ml_techniques_binary_var.loc["C.Difficile (Disc.)","SVM TPR"] = np.var(true_pos[tpok])
    ml_techniques_binary_var.loc["C.Difficile (Disc.)","SVM FPR"] = np.var(false_pos[fpok])
    ml_techniques_binary_var.loc["C.Difficile (Disc.)","SVM AUROC"] = np.var(aucrocs[arok])


    tot_miss = np.empty(N)
    true_pos = np.empty(N)
    false_pos = np.empty(N)
    tpok = np.empty(N,dtype = bool)
    fpok = np.empty(N,dtype= bool)
    aucrocs = np.empty(N)
    arok = np.zeros(N).astype(bool)
    for i in range(N):
        train_tp,test_tp = prep_data_discrete(samples)
        classifier = RandomForestClassifier()
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum(prediction != test_tp[1])/len(test_tp[1])
        tot_miss[i] = miss_prop
        if sum(test_tp[1] == 1):
            true_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==1) for j in range(len(prediction))])/sum(test_tp[1] == 1)
            tpok[i] = True
        else:
            tpok[i] = False
        if sum(test_tp[1] == 0):
            false_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==0) for j in range(len(prediction))])/sum(test_tp[1] == 0)
            fpok[i] = True
        else:
            fpok[i] = False
        y_score = classifier.predict_proba(test_tp[0])[:,1]
        try:
            aucrocs[i] = metrics.roc_auc_score(test_tp[1],y_score)
            arok[i]=True
        except:
            pass

 
    ml_techniques_binary_mean.loc["C.Difficile (Disc.)","RF Miss"] = np.mean(tot_miss)
    ml_techniques_binary_mean.loc["C.Difficile (Disc.)","RF TPR"] = np.mean(true_pos[tpok])
    ml_techniques_binary_mean.loc["C.Difficile (Disc.)","RF FPR"] = np.mean(false_pos[fpok])
    ml_techniques_binary_mean.loc["C.Difficile (Disc.)","RF AUROC"] = np.mean(aucrocs[arok])

    ml_techniques_binary_var.loc["C.Difficile (Disc.)","RF Miss"] = np.var(tot_miss)
    ml_techniques_binary_var.loc["C.Difficile (Disc.)","RF TPR"] = np.var(true_pos[tpok])
    ml_techniques_binary_var.loc["C.Difficile (Disc.)","RF FPR"] = np.var(false_pos[fpok])
    ml_techniques_binary_var.loc["C.Difficile (Disc.)","RF AUROC"] = np.var(aucrocs[arok])

    with open("friendlySamples/upto_species/bifido_species_dict_binary.pk","rb") as fl:
        samples = pk.load(fl)

    tot_miss = np.empty(N)
    true_pos = np.empty(N)
    false_pos = np.empty(N)
    tpok = np.empty(N,dtype = bool)
    fpok = np.empty(N,dtype= bool)
    aucrocs = np.empty(N)
    arok = np.zeros(N).astype(bool)
    for i in range(N):
        train_tp,test_tp = prep_data_continuous(samples)
        classifier = svm.SVC(probability = True)
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum(prediction != test_tp[1])/len(test_tp[1])
        tot_miss[i] = miss_prop
        if sum(test_tp[1] == 1):
            true_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==1) for j in range(len(prediction))])/sum(test_tp[1] == 1)
            tpok[i] = True
        else:
            tpok[i] = False
        if sum(test_tp[1] == 0):
            false_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==0) for j in range(len(prediction))])/sum(test_tp[1] == 0)
            fpok[i] = True
        else:
            fpok[i] = False
        y_score = classifier.predict_proba(test_tp[0])[:,1]
        try:
            aucrocs[i] = metrics.roc_auc_score(test_tp[1],y_score)
            arok[i]=True
        except:
            pass


    ml_techniques_binary_mean.loc["B.longum (Cont.)","SVM Miss"] = np.mean(tot_miss)
    ml_techniques_binary_mean.loc["B.longum (Cont.)","SVM TPR"] = np.mean(true_pos[tpok])
    ml_techniques_binary_mean.loc["B.longum (Cont.)","SVM FPR"] = np.mean(false_pos[fpok])
    ml_techniques_binary_mean.loc["B.longum (Cont.)","SVM AUROC"] = np.mean(aucrocs[arok])

    ml_techniques_binary_var.loc["B.longum (Cont.)","SVM Miss"] = np.var(tot_miss)
    ml_techniques_binary_var.loc["B.longum (Cont.)","SVM TPR"] = np.var(true_pos[tpok])
    ml_techniques_binary_var.loc["B.longum (Cont.)","SVM FPR"] = np.var(false_pos[fpok])
    ml_techniques_binary_var.loc["B.longum (Cont.)","SVM AUROC"] = np.var(aucrocs[arok])

    tot_miss = np.empty(N)
    true_pos = np.empty(N)
    false_pos = np.empty(N)
    tpok = np.empty(N,dtype = bool)
    fpok = np.empty(N,dtype= bool)
    aucrocs = np.empty(N)
    arok = np.zeros(N).astype(bool)
    for i in range(N):
        train_tp,test_tp = prep_data_continuous(samples)
        classifier = RandomForestClassifier()
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum(prediction != test_tp[1])/len(test_tp[1])
        tot_miss[i] = miss_prop
        if sum(test_tp[1] == 1):
            true_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==1) for j in range(len(prediction))])/sum(test_tp[1] == 1)
            tpok[i] = True
        else:
            tpok[i] = False
        if sum(test_tp[1] == 0):
            false_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==0) for j in range(len(prediction))])/sum(test_tp[1] == 0)
            fpok[i] = True
        else:
            fpok[i] = False
            y_score = classifier.predict_proba(test_tp[0])[:,1]
        try:
            aucrocs[i] = metrics.roc_auc_score(test_tp[1],y_score)
            arok[i]=True
        except:
            pass

  
    ml_techniques_binary_mean.loc["B.longum (Cont.)","RF Miss"] = np.mean(tot_miss)
    ml_techniques_binary_mean.loc["B.longum (Cont.)","RF TPR"] = np.mean(true_pos[tpok])
    ml_techniques_binary_mean.loc["B.longum (Cont.)","RF FPR"] = np.mean(false_pos[fpok])
    ml_techniques_binary_mean.loc["B.longum (Cont.)","RF AUROC"] = np.mean(aucrocs[arok])

    ml_techniques_binary_var.loc["B.longum (Cont.)","RF Miss"] = np.var(tot_miss)
    ml_techniques_binary_var.loc["B.longum (Cont.)","RF TPR"] = np.var(true_pos[tpok])
    ml_techniques_binary_var.loc["B.longum (Cont.)","RF FPR"] = np.var(false_pos[fpok])
    ml_techniques_binary_var.loc["B.longum (Cont.)","RF AUROC"] = np.var(aucrocs[arok])

    tot_miss = np.empty(N)
    true_pos = np.empty(N)
    false_pos = np.empty(N)
    tpok = np.empty(N,dtype = bool)
    fpok = np.empty(N,dtype= bool)
    aucrocs = np.empty(N)
    arok = np.zeros(N).astype(bool)
    for i in range(N):
        train_tp,test_tp = prep_data_discrete(samples)
        classifier = svm.SVC(probability = True)
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum(prediction != test_tp[1])/len(test_tp[1])
        tot_miss[i] = miss_prop
        if sum(test_tp[1] == 1):
            true_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==1) for j in range(len(prediction))])/sum(test_tp[1] == 1)
            tpok[i] = True
        else:
            tpok[i] = False
        if sum(test_tp[1] == 0):
            false_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==0) for j in range(len(prediction))])/sum(test_tp[1] == 0)
            fpok[i] = True
        else:
            fpok[i] = False
        y_score = classifier.predict_proba(test_tp[0])[:,1]
        try:
            aucrocs[i] = metrics.roc_auc_score(test_tp[1],y_score)
            arok[i]=True
        except:
            pass

  
    ml_techniques_binary_mean.loc["B.longum (Disc.)","SVM Miss"] = np.mean(tot_miss)
    ml_techniques_binary_mean.loc["B.longum (Disc.)","SVM TPR"] = np.mean(true_pos[tpok])
    ml_techniques_binary_mean.loc["B.longum (Disc.)","SVM FPR"] = np.mean(false_pos[fpok])
    ml_techniques_binary_mean.loc["B.longum (Disc.)","SVM AUROC"] = np.mean(aucrocs[arok])

    ml_techniques_binary_var.loc["B.longum (Disc.)","SVM Miss"] = np.var(tot_miss)
    ml_techniques_binary_var.loc["B.longum (Disc.)","SVM TPR"] = np.var(true_pos[tpok])
    ml_techniques_binary_var.loc["B.longum (Disc.)","SVM FPR"] = np.var(false_pos[fpok])
    ml_techniques_binary_var.loc["B.longum (Disc.)","SVM AUROC"] = np.var(aucrocs[arok])

    tot_miss = np.empty(N)
    true_pos = np.empty(N)
    false_pos = np.empty(N)
    tpok = np.empty(N,dtype = bool)
    fpok = np.empty(N,dtype= bool)
    aucrocs = np.empty(N)
    arok = np.zeros(N).astype(bool)
    for i in range(N):
        train_tp,test_tp = prep_data_discrete(samples)
        classifier = RandomForestClassifier()
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum(prediction != test_tp[1])/len(test_tp[1])
        tot_miss[i] = miss_prop
        if sum(test_tp[1] == 1):
            true_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==1) for j in range(len(prediction))])/sum(test_tp[1] == 1)
            tpok[i] = True
        else:
            tpok[i] = False
        if sum(test_tp[1] == 0):
            false_pos[i] = sum([(prediction[j]==1) and (test_tp[1][j]==0) for j in range(len(prediction))])/sum(test_tp[1] == 0)
            fpok[i] = True
        else:
            fpok[i] = False
        y_score = classifier.predict_proba(test_tp[0])[:,1]
        try:
            aucrocs[i] = metrics.roc_auc_score(test_tp[1],y_score)
            arok[i]=True
        except:
            pass

  
    ml_techniques_binary_mean.loc["B.longum (Disc.)","RF Miss"] = np.mean(tot_miss)
    ml_techniques_binary_mean.loc["B.longum (Disc.)","RF TPR"] = np.mean(true_pos[tpok])
    ml_techniques_binary_mean.loc["B.longum (Disc.)","RF FPR"] = np.mean(false_pos[fpok])
    ml_techniques_binary_mean.loc["B.longum (Disc.)","RF AUROC"] = np.mean(aucrocs[arok])

    ml_techniques_binary_var.loc["B.longum (Disc.)","RF Miss"] = np.var(tot_miss)
    ml_techniques_binary_var.loc["B.longum (Disc.)","RF TPR"] = np.var(true_pos[tpok])
    ml_techniques_binary_var.loc["B.longum (Disc.)","RF FPR"] = np.var(false_pos[fpok])
    ml_techniques_binary_var.loc["B.longum (Disc.)","RF AUROC"] = np.var(aucrocs[arok])





    with open("friendlySamples/upto_species/lacto_day0_human_species.pk","rb") as fl:
        samples = pk.load(fl)

    avgrmserror = np.empty(N)
    avg_kendall = np.empty(N)
    avg_spearman = np.empty(N)
    for i in range(N):
        train_tp,test_tp = prep_data_continuous(samples)
        classifier = svm.SVR()
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum((prediction - test_tp[1])**2)/len(test_tp[1])
        avgrmserror[i] = miss_prop
        avg_kendall[i],_ = kendalltau(prediction,test_tp[1])
        avg_spearman[i],_ = spearmanr(prediction,test_tp[1])
    ml_Lplantarum.loc["Continuous","SVM Avg Error"] = np.mean(avgrmserror)
    ml_Lplantarum.loc["Continuous","SVM Variance Error"] = np.var(avgrmserror)
    ml_Lplantarum.loc["Continuous","SVM Kendall"] = np.mean(avg_kendall)
    ml_Lplantarum.loc["Continuous","SVM Spearman"] = np.mean(avg_spearman)


    avgrmserror = np.empty(N)
    avg_kendall = np.empty(N)
    avg_spearman = np.empty(N)
    for i in range(N):
        train_tp,test_tp = prep_data_continuous(samples)
        classifier = RandomForestRegressor()
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum((prediction - test_tp[1])**2)/len(test_tp[1])
        avgrmserror[i] = miss_prop
        avg_kendall[i],_ = kendalltau(prediction,test_tp[1])
        avg_spearman[i],_ = spearmanr(prediction,test_tp[1])

    ml_Lplantarum.loc["Continuous","RF Avg Error"] = np.mean(avgrmserror)
    ml_Lplantarum.loc["Continuous","RF Variance Error"] = np.var(avgrmserror)
    ml_Lplantarum.loc["Continuous","RF Kendall"] = np.mean(avg_kendall)
    ml_Lplantarum.loc["Continuous","RF Spearman"] = np.mean(avg_spearman)

    avgrmserror = np.empty(N)
    avg_kendall = np.empty(N)
    avg_spearman = np.empty(N)
    for i in range(N):
        train_tp,test_tp = prep_data_discrete(samples)
        classifier = svm.SVR()
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum((prediction - test_tp[1])**2)/len(test_tp[1])
        avgrmserror[i] = miss_prop
        avg_kendall[i],_ = kendalltau(prediction,test_tp[1])
        avg_spearman[i],_ = spearmanr(prediction,test_tp[1])
   
    ml_Lplantarum.loc["Discrete","SVM Avg Error"] = np.mean(avgrmserror)
    ml_Lplantarum.loc["Discrete","SVM Variance Error"] = np.var(avgrmserror)
    ml_Lplantarum.loc["Discrete","SVM Kendall"] = np.mean(avg_kendall[~np.isnan(avg_kendall)])
    ml_Lplantarum.loc["Discrete","SVM Spearman"] = np.mean(avg_spearman[~np.isnan(avg_spearman)])

    avgrmserror = np.empty(N)
    avg_kendall = np.empty(N)
    avg_spearman = np.empty(N)
    for i in range(N):
        train_tp,test_tp = prep_data_discrete(samples)
        classifier = RandomForestRegressor()
        classifier.fit(*train_tp)
        prediction = classifier.predict(test_tp[0])
        miss_prop = sum((prediction - test_tp[1])**2)/len(test_tp[1])
        avgrmserror[i] = miss_prop
        avg_kendall[i],_ = kendalltau(prediction,test_tp[1])
        avg_spearman[i],_ = spearmanr(prediction,test_tp[1])

   
    ml_Lplantarum.loc["Discrete","RF Avg Error"] = np.mean(avgrmserror)
    ml_Lplantarum.loc["Discrete","RF Variance Error"] = np.var(avgrmserror)
    ml_Lplantarum.loc["Discrete","RF Kendall"] = np.mean(avg_kendall[~np.isnan(avg_kendall)])
    ml_Lplantarum.loc["Discrete","RF Spearman"] = np.mean(avg_spearman[~np.isnan(avg_spearman)])

    ml_Lplantarum.to_csv(os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","pb_design_results","ML","ml_lacto.csv"))