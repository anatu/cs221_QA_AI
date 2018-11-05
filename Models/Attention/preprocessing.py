from keras.preprocessing.sequence import pad_sequences
import os
import re
import numpy as np

class Preprocessor():
    # Pass in the path to the data being read
    def __init__(self):
        self.EMBED_HIDDEN_SIZE = 50
        self.SENT_HIDDEN_SIZE = 100
        self.QUERY_HIDDEN_SIZE = 100
        self.BATCH_SIZE = 32
        self.EPOCHS = 5
        self.GLOVE_PATH = "C:/users/Anand\ Natu/Desktop"
        self.EMBEDDING_DIM = 100
        print("Preprocessor + model dimensions and parameters initialized")
        print('Embedding Hidden Size = {}'.format(self.EMBED_HIDDEN_SIZE))
        print('Sentence Hidden Size = {}'.format(self.SENT_HIDDEN_SIZE))
        print('Batch Size = {}'.format(self.BATCH_SIZE))
        print('# of Training Epochs = {}'.format(self.EPOCHS))



        

    def vectorize_stories(self, data, word_idx, story_maxlen, query_maxlen):
        xs = []
        xqs = []
        ys = []
        for story, query, answer in data:
            x = [word_idx[w] for w in story]
            xq = [word_idx[w] for w in query]
            # let's not forget that index 0 is reserved
            #y = np.zeros(len(word_idx) + 1)
            #y[word_idx[answer]] = 1
            y = np.zeros((2, story_maxlen+3))
            if(answer == ['yes']):
                y[0][story_maxlen] = 1
                y[1][story_maxlen] = 1
            elif(answer == ['no']):
                y[0][story_maxlen+1] = 1
                y[1][story_maxlen+1] = 1
            elif(answer == ['null']):
                y[0][story_maxlen+2] = 1
                y[1][story_maxlen+2] = 1
            else:
                y[0][answer[0]] = 1
                y[1][answer[1]] = 1
            xs.append(x)
            xqs.append(xq)
            ys.append(y)
        return (pad_sequences(xs, maxlen=story_maxlen),
                pad_sequences(xqs, maxlen=query_maxlen), np.array(ys))


    # CMU QA Dataset 
    # Tokenize data for feeding into neural network
    def prepare_cmu_data(self, data_path):    
        # Read in the corpus
        corpus = {}
        for sets in range(1, 5):
            for a in range(1, 11):
                f = open(os.path.join(data_path, "data", "set{}".format(sets), "a{}.txt.clean".format(a)), 'r', encoding="ANSI")
                content = f.read()
                f.close()
                content = content.replace('\n', ' ').replace('\r', '')
                content = re.sub(r'[^a-zA-Z ]', '', content).lower().split()
                corpus["data/set{}/a{}".format(sets, a)] = content
        
        # Read in the question-answer pairs
        qa_file = open(os.path.join(data_path, "question_answer_pairs.txt"), 'r', encoding = "ANSI")
        content = qa_file.readlines()
        qa_file.close()
        content = [x.strip() for x in content]
        content = [re.split(r'\t+', x) for x in content]
        questions = []

        for line in content[1:]:
            if len(line) > 5:
                article = corpus[line[5]]
                q = re.sub(r'[^a-zA-Z ]', '', line[1]).lower().split()
                ans = re.sub(r'[^a-zA-Z ]', '', line[2]).lower().split()
                y = [-1, -1]
                #print(ans)
                for i in range(len(article)-len(ans)):
                    #print(article[i:i+len(ans)])
                    if(article[i:i+len(ans)] == ans):
                        y = [i, i+len(ans)]
                        #print('Found one!')
                if(y == [-1, -1]):
                    if(ans[0].lower() == 'yes'):
                        y = ['yes']
                    if(ans[0].lower() == 'no'):
                        y = ['no']
                    if(ans[0].lower() == 'null'):
                        y = ['null']                        
                questions.append((article, q, y))
        
        np.random.shuffle(questions)
        train, test = questions[:1020], questions[1020:]
        
        self.vocab = set()
        for story, q, answer in train + test:
            words = story + q
            self.vocab |= set(words)
        self.vocab = sorted(self.vocab)

        self.word_idx = dict((c, i + 1) for i, c in enumerate(self.vocab))
        vocab_size = len(self.vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in train + test)))
        query_maxlen = max(map(len, (x for _, x, _ in train + test)))

        x, xq, y = self.vectorize_stories(train, self.word_idx, story_maxlen, query_maxlen)
        tx, txq, ty = self.vectorize_stories(test, self.word_idx, story_maxlen, query_maxlen)

        return x, tx, xq, txq, y, ty

    def generate_embedding_matrix(self, word_idx, glove_dir):
        f = open(os.path.join(self.GLOVE_PATH, glove_dir, "glove.6B.{}d.txt".format(self.EMBEDDING_DIM)), 'r', encoding = "ANSI")
        embeddings_index = {}
        for line in f:
            values = line.split(" ")
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError:
                print(values[1:])
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((len(word_idx) + 1, self.EMBEDDING_DIM))
        for word, i in word_idx.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        print('Found %s word vectors.' % len(embeddings_index))
        return embedding_matrix



# FOR TESTING PURPOSES ONLY
if __name__ == "__main__":
    pp = Preprocessor()
    x, tx, xq, txq, y, ty = pp.prepare_cmu_data("../../Data/Question_Answer_Dataset_v1.2/S08")
    embedding_matrix = pp.generate_embedding_matrix(pp.word_idx, r"C:\Users\Anand Natu\Desktop\glove.6B")
    print(embedding_matrix)