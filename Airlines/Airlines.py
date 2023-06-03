import pandas as pd
from sklearn.model_selection import StratifiedKFold

file = pd.read_csv("airlanes.csv", encoding="latin", names=["ID", "Class", "SC", "NR", "NRC", "Airlane", "SG", "Name", "NRG", "Retweets", "Text", "TC", "Date", "Location", "TZ"])
file = file.sample(frac=1)

tweets=file['Text'].tolist()
clases=file['Class'].tolist()

def division(tweets, clases, k):
    skf = StratifiedKFold(n_splits=k)
    i = 1
    train_neg_1, train_pos_1, test_neg_1, test_pos_1 = [], [], [], []
    train_neg_2, train_pos_2, test_neg_2, test_pos_2 = [], [], [], []
    train_neg_3, train_pos_3, test_neg_3, test_pos_3 = [], [], [], []
    train_neg_4, train_pos_4, test_neg_4, test_pos_4 = [], [], [], []
    train_neg_5, train_pos_5, test_neg_5, test_pos_5 = [], [], [], []
    for train, test in skf.split(tweets, clases):
        if i == 1:
            for t in train:
                clase = clases[t]
                if clase == 'negative':
                    train_neg_1.append(tweets[t])
                elif clase == 'positive':
                    train_pos_1.append(tweets[t])
            for t in test:
                clase = clases[t]
                if clase == 'negative':
                    test_neg_1.append(tweets[t])
                elif clase == 'positive':
                    test_pos_1.append(tweets[t])
        elif i == 2:
            for t in train:
                clase = clases[t]
                if clase == 'negative':
                    train_neg_2.append(tweets[t])
                elif clase == 'positive':
                    train_pos_2.append(tweets[t])
            for t in test:
                clase = clases[t]
                if clase == 'negative':
                    test_neg_2.append(tweets[t])
                elif clase == 'positive':
                    test_pos_2.append(tweets[t])
        elif i == 3:
            for t in train:
                clase = clases[t]
                if clase == 'negative':
                    train_neg_3.append(tweets[t])
                elif clase == 'positive':
                    train_pos_3.append(tweets[t])
            for t in test:
                clase = clases[t]
                if clase == 'negative':
                    test_neg_3.append(tweets[t])
                elif clase == 'positive':
                    test_pos_3.append(tweets[t])
        elif i == 4:
            for t in train:
                clase = clases[t]
                if clase == 'negative':
                    train_neg_4.append(tweets[t])
                elif clase == 'positive':
                    train_pos_4.append(tweets[t])
            for t in test:
                clase = clases[t]
                if clase == 'negative':
                    test_neg_4.append(tweets[t])
                elif clase == 'positive':
                    test_pos_4.append(tweets[t])
        elif i == 5:
            for t in train:
                clase = clases[t]
                if clase == 'negative':
                    train_neg_5.append(tweets[t])
                elif clase == 'positive':
                    train_pos_5.append(tweets[t])
            for t in test:
                clase = clases[t]
                if clase == 'negative':
                    test_neg_5.append(tweets[t])
                elif clase == 'positive':
                    test_pos_5.append(tweets[t])
        i+=1

    return train_neg_1, train_pos_1, test_neg_1, test_pos_1, train_neg_2, train_pos_2, test_neg_2, test_pos_2, train_neg_3, train_pos_3, test_neg_3, test_pos_3, train_neg_4, train_pos_4, test_neg_4, test_pos_4, train_neg_5, train_pos_5, test_neg_5, test_pos_5
        

def write(file,train_neg,train_pos,test_neg,test_pos) :
    i = 0  
    for tweet in train_neg:
        f = open(file + "/train/neg/tweet" + str(i) + '.txt', "a+", encoding="utf-8")
        f.write(tweet)
        f.close()
        i += 1

    i = 0  
    for tweet in train_pos:
        f = open(file + "/train/pos/tweet" + str(i) + '.txt', "a+", encoding="utf-8")
        f.write(tweet)
        f.close()
        i += 1

    i = 0  
    for tweet in test_neg:
        f = open(file + "/test/neg/tweet" + str(i) + '.txt', "a+", encoding="utf-8")
        f.write(tweet)
        f.close()
        i += 1

    i = 0  
    for tweet in test_pos:
        f = open(file + "/test/pos/tweet" + str(i) + '.txt', "a+", encoding="utf-8")
        f.write(tweet)
        f.close()
        i += 1

train_neg_1, train_pos_1, test_neg_1, test_pos_1, train_neg_2, train_pos_2, test_neg_2, test_pos_2, train_neg_3, train_pos_3, test_neg_3, test_pos_3, train_neg_4, train_pos_4, test_neg_4, test_pos_4, train_neg_5, train_pos_5, test_neg_5, test_pos_5 =  division(tweets, clases, 5)



print(len(train_neg_1), len(train_pos_1), len(test_neg_1), len(test_pos_1))
print(len(train_neg_2), len(train_pos_2), len(test_neg_2), len(test_pos_2))
print(len(train_neg_3), len(train_pos_3), len(test_neg_3), len(test_pos_3))
print(len(train_neg_4), len(train_pos_4), len(test_neg_4), len(test_pos_4))
print(len(train_neg_5), len(train_pos_5), len(test_neg_5), len(test_pos_5))
#write('d1', train_neg_1, train_pos_1, test_neg_1, test_pos_1)
#write('d2', train_neg_2, train_pos_2, test_neg_2, test_pos_2)
#write('d3', train_neg_3, train_pos_3, test_neg_3, test_pos_3)
#write('d4', train_neg_4, train_pos_4, test_neg_4, test_pos_4)
#write('d5', train_neg_5, train_pos_5, test_neg_5, test_pos_5)