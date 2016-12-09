import csv
import os
import shutil
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np
# import matplotlib.pyplot as plt
import math
import nltk
import codecs
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle
from multiprocessing import Pool
from classification import classification
# from gensim.models.word2vec import Word2Vec
from sklearn.metrics import roc_curve, auc
from multiprocessing import Process
# from Word2VecExtraction import MeanEmbeddingVectorizer,TfidfEmbeddingVectorizer
# from gensim.models.word2vec import Word2Vec
from nltk import ngrams
from collections import Counter
filename="HN_posts_year_to_Sep_26_2016.csv"

# load data.csv to a dict
def Load_Data_Dict(filename):
    data_dict={}
    with open(filename,'rb') as csvfile:
        reader=csv.DictReader(csvfile)
        field_names=reader.fieldnames
        for row in reader:
            data_dict[row['id']]=row
    return data_dict,field_names

# load data.csv to a list
def Load_Data_List(filename):
    data_list=[]
    with open(filename,'rb') as csvfile:
        reader=csv.DictReader(csvfile)
        for row in reader:
            data_list.append(row)
    return data_list

# Generate data.csv in a given directory using the data dict of HN_posts_year_to_Sep_26_2016.csv
def Gen_Csv_Data(dir_name,data_dict,field_names):
    with open(dir_name+'data.csv','w') as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=field_names)
        writer.writeheader()
        files=os.listdir(dir_name)
        for f in files:
            if f.endswith('.txt'):
                # print('archiving ' + f)
                id=f[0:len(f)-4]
                writer.writerow(data_dict[id])


# Tokeninzer: take the string of the document and return the list of tokens (list of unicode strings)
def Customized_Tokenizer(d):
    tokenizer = RegexpTokenizer(r'[a-zA-Z#\+]+')
    tokens=tokenizer.tokenize(d.lower())
    st = LancasterStemmer()
    # tokens=[codecs.decode(t,'utf-8','ignore') for t in tokens]
    result_tokens=[]
    for t in tokens:
        ut=codecs.decode(t,'utf-8','ignore')
        if ut in stopwords.words("english"):
            continue
        if len(ut)<3:
            continue
        result_tokens.append(st.stem(ut))
    return result_tokens

def id_read(filename):
    with open(filename, 'rb') as textfile:
        return str(textfile.read())

# Take directory name and return a list of document in strings
def Load_Text_Data(dirname):
    id_list = []
    with open(dirname+'data.csv','rb') as csvfile:
        reader=csv.DictReader(csvfile);
        field_names=reader.fieldnames;
        for row in reader:
            id_list.append(str(row['id']))

    filenames=map(lambda id : dirname+id+'.txt', id_list)
    return map(id_read,filenames)

# Tokenize training data and write train_counts.plk and count_vec.plk into the folder
def Tokenize_Train_Data(dirname,train_data):
    count_vec=CountVectorizer(tokenizer=Customized_Tokenizer, ngram_range=(2,2))
    train_counts=count_vec.fit_transform(train_data)
    print(train_counts.shape)
    with open(dirname+'train_counts.plk','wb') as f:
        pickle.dump(train_counts, f, pickle.HIGHEST_PROTOCOL)

    with open(dirname+'count_vec.plk','wb') as f:
        pickle.dump(count_vec, f, pickle.HIGHEST_PROTOCOL)

# Tokenize test data and write test_counts.plk into the folder
def Tokenize_Test_Data(dirname,test_data,count_vec):
    test_counts=count_vec.transform(test_data)
    print(test_counts.shape)
    with open(dirname+'test_counts.plk','wb') as f:
        pickle.dump(test_counts, f, pickle.HIGHEST_PROTOCOL)

# Write pickle data
def Write_Pickle_Data(data,filename):
    with open(filename,'wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

# Load data ending with '.plk'
def Load_Pickle_Data(filename):
    with open(filename,'rb') as f:
        data=pickle.load(f)
        return data

def Get_Tokens(dirname):
    print('Get_Tokens:'+dirname)
    token_path=dirname+'tokens.plk'
    if os.path.isfile(token_path):
        tokens=Load_Pickle_Data(token_path)
    else:
        p=Pool()
        data=Load_Text_Data(dirname)
        tokens=p.map(Customized_Tokenizer,data)
        Write_Pickle_Data(tokens,dirname+'tokens.plk')
    return tokens

def Get_Lables(dirname,popular_thresh):
    data_list=Load_Data_List(dirname+'data.csv')
    return [1 if int(row['num_points'])>popular_thresh else 0 for row in data_list]

def parallel_classification(Xtrain,ytrain,Xtest,ytest,pname,pclass_weight={}):
    p=Process(target=classification,args=(Xtrain,ytrain,Xtest,ytest,pname,pclass_weight))
    p.start()
    p.join()

def under_sample(X,y):
    train_1=[]
    train_0=[]
    num_train=len(y)
    num_1=0
    for i in range(num_train):
        if y[i] == 1:
            train_1.append(i)
            num_1+=1
        else:
            train_0.append(i)
    under_sampled=random.sample(train_0,num_1)
    X_indices=train_1+under_sampled
    X_resampled=X[X_indices]
    y_resampled=[y[i] for i in X_indices]
    return X_resampled, y_resampled

def over_sample(X,y):
    train_1=[]
    train_0=[]
    num_train=len(y)
    num_1=0
    num_0=0
    for i in range(num_train):
        if y[i] == 1:
            train_1.append(i)
            num_1+=1
        else:
            train_0.append(i)
            num_0+=1
    over_sampled=[train_1[random.randint(0,num_1-1)] for i in range(num_0)]
    X_indices=train_0+over_sampled
    X_resampled=X[X_indices]
    y_resampled=[y[i] for i in X_indices]
    return X_resampled, y_resampled

def train():
    train_folder='train_140/'
    test_folder='test_60/'
    popular_thresh=30
    data_list=Load_Data_List(train_folder+'/data.csv')
    num_train=len(data_list)
    train_targets=[1 if int(row['num_points'])>popular_thresh else 0 for row in data_list]
    data_list=Load_Data_List(test_folder+'/data.csv')
    num_test=len(data_list)
    test_targets=[1 if int(row['num_points'])>popular_thresh else 0 for row in data_list]
    num_popular_train=sum(train_targets)
    num_popular_test=sum(test_targets)
    print('num_train:'+str(num_train)+',num_popular_train:'+str(num_popular_train)+',ratio:'+str(float(num_popular_train)/num_train))
    print('num_test:' + str(num_test)+',num_popular_test:'+str(num_popular_test)+',ratio:'+str(float(num_popular_test)/num_test))
    train_counts=Load_Pickle_Data(train_folder+'train_counts.plk')
    test_counts=Load_Pickle_Data(test_folder+'test_counts.plk')
    parallel_classification(train_counts, train_targets, test_counts, test_targets, 'simple_counts')
    # parallel_classification(train_counts, train_targets, test_counts, test_targets, 'simple_counts_class_weights_1-10', {0: 1, 1: 10})
    # parallel_classification(train_counts, train_targets, test_counts, test_targets, 'simple_counts_class_weights_1-50', {0: 1, 1: 50})

    tf_transformer=TfidfTransformer()
    train_tfidf = tf_transformer.fit_transform(train_counts)
    print(train_tfidf.shape)
    test_tfidf=tf_transformer.transform(test_counts)
    print(test_tfidf.shape)
    parallel_classification(train_tfidf,train_targets,test_tfidf,test_targets,'tfidf')
    # parallel_classification(train_tfidf, train_targets, test_tfidf, test_targets, 'tfidf_class_weights_1-10',{0:1,1:10})
    # parallel_classification(train_tfidf, train_targets, test_tfidf, test_targets, 'tfidf_class_weights_1-50',{0:1,1:50})


def N_Gram_Tokenize_Train_Data(dirname, n):
    train_tokens = Get_Tokens(dirname)
    train_grams = map(lambda tokens: ngrams(tokens, n), train_tokens)
    train_n_gram_feature = map(lambda grams: Counter(grams), train_grams)
    dict_vec = DictVectorizer()
    train_counts = dict_vec.fit_transform(train_n_gram_feature)

    with open(dirname+str(n)+'-gram_train_counts.plk','wb') as f:
        pickle.dump(train_counts, f, pickle.HIGHEST_PROTOCOL)

    with open(dirname+str(n)+'-gram_dict_vec.plk','wb') as f:
        pickle.dump(dict_vec, f, pickle.HIGHEST_PROTOCOL)


def N_Gram_Tokenize_Test_Data(dirname, dict_vec, n):
    test_tokens = Get_Tokens(dirname)
    test_grams = map(lambda tokens: ngrams(tokens, n), test_tokens)
    test_n_gram_feature = map(lambda grams: Counter(grams), test_grams)
    test_counts = dict_vec.transform(test_n_gram_feature)

    with open(dirname+str(n)+'-gram_test_counts.plk','wb') as f:
        pickle.dump(test_counts, f, pickle.HIGHEST_PROTOCOL)

def train_n_gram(n):
    train_folder='train_original/'
    test_folder='test_original/'
    popular_thresh=30
    data_list=Load_Data_List(train_folder+'/data.csv')
    num_train=len(data_list)
    train_targets=[1 if int(row['num_points'])>popular_thresh else 0 for row in data_list]
    data_list=Load_Data_List(test_folder+'/data.csv')
    num_test=len(data_list)
    test_targets=[1 if int(row['num_points'])>popular_thresh else 0 for row in data_list]
    num_popular_train=sum(train_targets)
    num_popular_test=sum(test_targets)
    print('num_train:'+str(num_train)+',num_popular_train:'+str(num_popular_train)+',ratio:'+str(float(num_popular_train)/num_train))
    print('num_test:' + str(num_test)+',num_popular_test:'+str(num_popular_test)+',ratio:'+str(float(num_popular_test)/num_test))

    train_counts = Load_Pickle_Data(train_folder+str(n)+'-gram_train_counts.plk')
    test_counts = Load_Pickle_Data(test_folder+str(n)+'-gram_test_counts.plk')
    # name = str(n)+'-gram_simple_counts'
    # parallel_classification(train_counts, train_targets, test_counts, test_targets, name)
    #
    # X_resampled, y_resampled = under_sample(train_counts, train_targets)
    # report_name = name + '-undersample'
    # parallel_classification(X_resampled, y_resampled, test_counts, test_targets, report_name)
    #
    # X_resampled, y_resampled = over_sample(train_counts, train_targets)
    # report_name = name + '-oversample'
    # parallel_classification(X_resampled, y_resampled, test_counts, test_targets, report_name)

    tf_transformer = TfidfTransformer()
    train_tfidf = tf_transformer.fit_transform(train_counts)
    test_tfidf = tf_transformer.transform(test_counts)
    name = str(n)+'-gram_tfidf'
    # parallel_classification(train_tfidf, train_targets, test_tfidf, test_targets, name)

    X_resampled, y_resampled = under_sample(train_tfidf, train_targets)
    report_name = name + '-undersample'
    parallel_classification(X_resampled, y_resampled, test_tfidf, test_targets, report_name)

    # X_resampled, y_resampled = over_sample(train_tfidf, train_targets)
    # report_name = name + '-oversample'
    # parallel_classification(X_resampled, y_resampled, test_tfidf, test_targets, report_name)

def w2v_train(vsize):
    train_folder='train_original/'
    test_folder='test_original/'
    popular_thresh=30 
    models=[('Word2Vec_Mean',MeanEmbeddingVectorizer),
            ('Word2Vec_Tfidf',TfidfEmbeddingVectorizer)]
    train_tokens=Get_Tokens(train_folder)
    test_tokens=Get_Tokens(test_folder)
    # train
    model=Word2Vec(train_tokens, size=vsize, window=5, min_count=5, workers=10)
    model.index2word
    w2v = {w: vec for w, vec in zip(model.index2word, model.syn0)}
    Write_Pickle_Data(w2v,train_folder+'w2v.plk')
    for name,vectorizer in models:
        v=vectorizer(w2v)
        v.fit(train_tokens,0)
        train_vecs=v.transform(train_tokens)
        print(train_vecs.shape)
        name='vsize='+str(vsize)+'-'+name
        Write_Pickle_Data(train_vecs,train_folder+name+'-train_vecs.plk')
        test_vecs=v.transform(test_tokens)
        print(test_vecs.shape)
        Write_Pickle_Data(test_vecs,test_folder+name+'-test_vecs.plk')
        train_targets=Get_Lables(train_folder,popular_thresh)
        test_targets=Get_Lables(test_folder,popular_thresh)
        report_name=name+'-regular'
        parallel_classification(train_vecs,train_targets,test_vecs,test_targets,report_name)
        X_resampled,y_resampled=under_sample(train_vecs,train_targets)
        report_name=name+'-undersample'
        parallel_classification(X_resampled,y_resampled,test_vecs,test_targets,report_name)
        X_resampled,y_resampled=over_sample(train_vecs,train_targets)
        report_name=name+'-oversample'
        parallel_classification(X_resampled,y_resampled,test_vecs,test_targets,report_name)



if __name__ == '__main__':
    # # Generate data.csv if needed
    # data_dict,field_names=Load_Data_Dict(filename)
    # Gen_Csv_Data('train_140/',data_dict,field_names) # Never miss '/' in the directory name
    # Gen_Csv_Data('test_60/',data_dict,field_names)

    # N-gram Tokenize
    # train_folder = 'train_original/'
    # test_folder = 'test_original/'
    # n_gram_min = 7
    # n_gram_max = 9

    # for n in range(n_gram_min, n_gram_max+1):
    #     N_Gram_Tokenize_Train_Data(train_folder, n)
    #     dict_vec = Load_Pickle_Data(train_folder+str(n)+'-gram_dict_vec.plk')
    #     N_Gram_Tokenize_Test_Data(test_folder, dict_vec, n)


    # # classification
    # # train()
    # # w2v_train(100)
    # # w2v_train(200)
    # # w2v_train(300)
    # # w2v_train(500)
    # # w2v_train(400)
    for n in range(n_gram_min, n_gram_max+1):
        train_n_gram(n)
    



