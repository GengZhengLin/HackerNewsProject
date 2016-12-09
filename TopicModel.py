from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

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
    # tfidf_transformer=TfidfTransformer()
    # tfidf=tfidf_transformer.fit_transform(tf)
    # nmf = NMF(n_components=n_topics, random_state=1,
    #           alpha=.1, l1_ratio=.5).fit(tfidf)
    # report_name='popular-nmf_topics-'+str(n_topics)+'.txt'
    # print_top_words(nmf, feature_names, n_top_words, report_name) 
    # topic_transformed = nmf.transform(tfidf)
    lda = LatentDirichletAllocation(n_topics=n_topics)
    lda.fit(tf)
    report_name='popular-lda_topics-'+str(n_topics)+'.txt'
    print_top_words(lda, feature_names, n_top_words, report_name)
    topic_transformed = lda.transform(tf)
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
