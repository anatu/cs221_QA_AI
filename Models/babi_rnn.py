from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen, answer_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = [word_idx[w] for w in answer]
        # let's not forget that index 0 is reserved
        #y = np.zeros(len(word_idx) + 1)
        #y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return (pad_sequences(xs, maxlen=story_maxlen),
            pad_sequences(xqs, maxlen=query_maxlen), pad_sequences(ys, maxlen=answer_maxlen))

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 1
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
                                                           QUERY_HIDDEN_SIZE))

#Read in CMU Question Answer Dataset
print('Reading in CMU dataset....')
print('Reading in corpus...')
corpus = {}
for sets in range(1, 7):
    for a in range(1, 11):
        f = open("..//Data//Question_Answer_Dataset_v1.2//S10//data//set{}//a{}.txt.clean".format(sets, a), 'r', encoding="ANSI")
        content = f.read()
        f.close()
        content = content.replace('\n', ' ').replace('\r', '')
        content = re.sub(r'[^a-zA-Z ]', '', content).lower().split()
        corpus["data/set{}/a{}".format(sets, a)] = content
#print(corpus['data/set1/a1'])

print('Reading in questions...')
f = open("..//Data//Question_Answer_Dataset_v1.2//S10//question_answer_pairs.txt", 'r', encoding="ANSI")
content = f.readlines()
f.close()
content = [x.strip() for x in content]
content = [re.split(r'\t+', x) for x in content]
questions = []
for line in content[1:]:
    if len(line) > 5:
        questions.append((corpus[line[5]], re.sub(r'[^a-zA-Z ]', '', line[1]).lower().split(), re.sub(r'[^a-zA-Z ]', '', line[2]).lower().split()))
np.random.shuffle(questions)
train, test = questions[:1020], questions[1020:]


vocab = set()
for story, q, answer in train + test:
    words = story + q + answer
    vocab |= set(words)
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))
answer_maxlen = max(map(len, (x for _, _, x in train + test)))

x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen, answer_maxlen)
tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen, answer_maxlen)



print('vocab = {}'.format(vocab))
print('x.shape = {}'.format(x.shape))
print('xq.shape = {}'.format(xq.shape))
print('y.shape = {}'.format(y.shape))
print('story_maxlen, query_maxlen, answer_maxlen = {}, {}, {}'.format(story_maxlen, query_maxlen, answer_maxlen))

print('Build model...')

sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

question = layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = layers.Dropout(0.3)(encoded_question)
encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

merged = layers.add([encoded_sentence, encoded_question])
merged = RNN(EMBED_HIDDEN_SIZE)(merged)
merged = layers.Dropout(0.3)(merged)
preds = layers.Dense(answer_maxlen, activation='relu')(merged)



model = Model([sentence, question], preds)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

print('Training')
model.fit([x, xq], y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)
loss, acc = model.evaluate([tx, txq], ty,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

print('Predicitng first training value: ')

ans = model.predict([x, xq][:10])

print('Question vector: ' + str(xq[0]))
q = [vocab[xq[i]] for i in range(len(xq[0]))]
print('Question: ' + ' '.join(q))


print('Model Answer vector: ' + str(ans[0]))
pred_ans = [vocab[round(ans[0][i])] for i in range(len(ans[0])) if round(ans[0][i]) >= 0 and round(ans[0][i]) < len(vocab)]
print('Model Answer vector: ' + ' '.join(pred_ans))

print('Correct Answer vector: ' + str(y[0]))
true_ans = pred_ans = [vocab[word_idx[y[i]]] for i in range(len(y))]

acc = sum([1 for i in ans[0] if round(ans[0]) in y[0] and i != 0])/sum([1 for i in y[0] if i != 0])
print('Accuracy is ' + str(acc))

accs = []
for i in range(len(ans)):
    accs.append(sum([1 for j in ans[i] if round(ans[i]) in y[i] and j != 0])/float(sum([1 for j in y[i] if j != 0])))
av_acc = np.mean(accs)

print('Average Training accuracy was ' + av_acc)

tans = model.predict([tx, txq][:10])
taccs = []
for i in range(len(tans)):
    taccs.append(sum([1 for j in tans[i] if round(tans[i]) in ty[i] and j != 0])/float(sum([1 for j in ty[i] if j != 0])))
tav_acc = np.mean(taccs)

print('Average test accuracy was ' + tav_acc)
