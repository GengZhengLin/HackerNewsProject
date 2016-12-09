import numpy as np
import matplotlib.pyplot as plt

def Extract_Report_Data(filename):
    with open(filename,'r') as f:
        data=[]
        line=f.readline()
        while line: 
            w=line.split()
            if len(w)>0 and w[0]=='precision':
                classifier_data=[]
                f.readline() # pass an empty line
                line=f.readline()
                numbers=line.split()[1:]
                for i in numbers:
                    classifier_data.append(float(i))
                line=f.readline()
                numbers=line.split()[1:]
                for i in numbers:
                    classifier_data.append(float(i))
                data.append(classifier_data)
            line=f.readline()
        return data

def Get_Colume(report_data,index):
    x=[]
    for data in report_data:
        x.append(data[index])
    return x

def Plot_Report_Data(report_data,classifier_names=[]):
    c0_precision=0
    c0_recall=1
    c1_precision=4
    c1_recall=5

    c0_precision_data=Get_Colume(report_data,c0_precision)
    c0_recall_data=Get_Colume(report_data,c0_recall)
    c1_precision_data=Get_Colume(report_data,c1_precision)
    c1_recall_data=Get_Colume(report_data,c1_recall)
    print c1_precision_data
    fig,axs=plt.subplots()
    ind=np.arange(len(report_data))
    print(ind)
    width=0.35    
    rects1=axs.bar(ind,c1_precision_data,width)
    axst=axs.twinx()
    rects2=axst.bar(ind+width,c1_recall_data,width,color='y')
    axs.set_xticklabels(classifier_names)
    plt.title('Class 1')
    axs.legend((rects1[0], rects2[0]), ('Precision', 'Recall'))

    plt.show()



if __name__ == '__main__':
    report_data=Extract_Report_Data('reports/2-gram_simple_counts-oversample-report.txt')
    classifier_names=['Ridge','Perceptron','Passive-Agressive','Linear (L2)','SVM (L2)', 'Linear (L1)', 'SVM (L1)', 'SVM (Elastic-Net)', 'MultinomialNB', 'BernoulliNB', 'Feature Selection (L1)'] # '' means ignore the classifier
    Plot_Report_Data(report_data,classifier_names)

