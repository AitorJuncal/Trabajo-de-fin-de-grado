train_neg = []
train_pos = []
test_neg = []
test_pos = []

with open('tweets.txt') as f:
    tweets = f.readlines()

with open('division.txt') as f:
    division = f.readlines()

for div in division:
    div = str(div).split()
    index = int(div[0])
    if div[1] == 'train' and div[2] == '0':
        train_neg.append(tweets[index])
    elif div[1] == 'train' and div[2] == '1':
        train_pos.append(tweets[index])
    elif div[1] == 'test' and div[2] == '0':
        test_neg.append(tweets[index])
    elif div[1] == 'test' and div[2] == '1':
        test_pos.append(tweets[index])



i = 0   
for tweet in train_neg:
    f = open("train/neg/tweet" + str(i) + '.txt', "a+")
    f.write(tweet)
    f.close()
    i += 1

i = 0   
for tweet in train_pos:
    f = open("train/pos/tweet" + str(i) + '.txt', "a+")
    f.write(tweet)
    f.close()
    i += 1

i = 0   
for tweet in test_neg:
    f = open("test/neg/tweet" + str(i) + '.txt', "a+")
    f.write(tweet)
    f.close()
    i += 1

i = 0   
for tweet in test_pos:
    f = open("test/pos/tweet" + str(i) + '.txt', "a+")
    f.write(tweet)
    f.close()
    i += 1

