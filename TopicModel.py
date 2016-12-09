from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from DataProcess import Load_Pickle_Data, Load_Data_List, Customized_Tokenizer

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words,filename):
    f=open(filename,'wb')
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        f.write("Topic #%d:i\n" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        f.write(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])+'\n\n')
    print()

def Extract_Topic_Model(tf,feature_names,num_topics=10):
    n_topics=num_topics
    # Fit the NMF model
    # print("Fitting the NMF model with tf-idf features, "
    #       "n_samples=%d and n_features=%d..."
    #       % (n_samples, n_features))
    # t0 = time()
    tfidf_transformer=TfidfTransformer()
    tfidf=tfidf_transformer.fit_transform(tf)
    nmf = NMF(n_components=n_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    report_name='popular-nmf_topics-'+str(n_topics)+'.txt'
    print_top_words(nmf, feature_names, n_top_words, report_name) 
    topic_transformed = nmf.transform(tfidf)
    # lda = LatentDirichletAllocation(n_topics=n_topics)
    # lda.fit(tf)
    # report_name='popular-lda_topics-'+str(n_topics)+'.txt'
    # print_top_words(lda, feature_names, n_top_words, report_name)
    # topic_transformed = lda.transform(tf)
    return topic_transformed
    
    # nmf = NMF(n_components=n_topics, random_state=1,
    #           alpha=.1, l1_ratio=.5).fit(tfidf)
    # print("done in %0.3fs." % (time() - t0))
    # 
    # print("\nTopics in NMF model:")
    # tfidf_feature_names = tf_vectorizer.get_feature_names()
    # report_name='nmf_topics-'+str(n_topics)+'.txt'
    # print_top_words(nmf, tfidf_feature_names, n_top_words, report_name)
    
    # lda = LatentDirichletAllocation(n_topics=n_topics)

    # t0 = time()
    # lda.fit(tf)
    # print("done in %0.3fs." % (time() - t0))
    # 
    # print("\nTopics in LDA model:")
    # tf_feature_names = tf_vectorizer.get_feature_names()
    # report_name='lda_topics-'+str(n_topics)+'.txt'
    # print_top_words(lda, tf_feature_names, n_top_words, report_name)

def Clean_Counts(train_counts,feature_names,lower_thresh=5,upper_thresh=1000):
    total_counts=(train_counts!=0).sum(0)
    ids=[]
    total_counts=total_counts.tolist()[0]
    print(len(total_counts))
    num_names=len(feature_names)
    for i in range(num_names):
        if total_counts[i] > lower_thresh and total_counts[i] < upper_thresh:
            ids.append(i)
    new_feature_names=[feature_names[i] for i in ids]
    return train_counts[:,ids],new_feature_names

def Get_Popular_Doc(train_counts,data_list,popular_thresh=30):
    ids=[]
    new_data_list=[]
    for i in range(len(data_list)):
        info=data_list[i]
        if int(info['num_points'])>popular_thresh:
            ids.append(i)
            new_data_list.append(info)
    return train_counts[ids],new_data_list

def Get_Info_Cluster(topic_transformed,data_list,get_min=True):
    num_clusters=topic_transformed.shape[1]
    clusters=[[] for i in range(num_clusters)]
    for i in range(topic_transformed.shape[0]):
        doc=topic_transformed[i]
        if get_min:
            topic=doc.argmin()
        else:
            topic=doc.argmax()
        clusters[topic].append(data_list[i])
    return clusters

def Write_Info_Clusters(info_clusters,data_list,filename):
    with open(filename,'w') as f:
        for i in range(len(info_clusters)):
            cluster=info_clusters[i]
            num_documents=len(cluster)
            total_points=0
            f.write('-'*20+'Topic '+str(i)+'-'*20+'\n')
            for info in cluster:
                f.write(info['id']+' '+info['title']+'\n')
                f.write(info['num_points']+' '+info['num_comments']+'\n')
                f.write(info['url']+'\n')
                f.write('\n')
                total_points+=int(info['num_points'])
            avg_points=float(total_points)/num_documents;
            f.write('num_documents:'+str(num_documents)+'avg_points:'+str(avg_points)+'\n')
            f.write('\n')

def Execute_Topic_Model(train_counts,data_list,feature_names):
    print(train_counts.shape)
    train_counts,feature_names=Clean_Counts(train_counts,feature_names,5,4000)
    train_counts,data_list=Get_Popular_Doc(train_counts,data_list,15)
    print(train_counts.shape)
    print(len(data_list))
    print(len(feature_names))
    # train_counts,feature_names=Clean_Counts(train_counts,feature_names)
    topic_transformed=Extract_Topic_Model(train_counts,feature_names)
    info_clusters=Get_Info_Cluster(topic_transformed,data_list,False)
    Write_Info_Clusters(info_clusters,data_list,'nmf-topic-infos.txt')


    
if __name__ == '__main__':
    train_folder='train_original_large/'
    train_counts=Load_Pickle_Data(train_folder+'train_counts.plk')
    count_vec=Load_Pickle_Data(train_folder+'count_vec.plk')
    feature_names=count_vec.get_feature_names()
    data_list=Load_Data_List(train_folder+'data.csv')
    Execute_Topic_Model(train_counts,data_list,feature_names)
