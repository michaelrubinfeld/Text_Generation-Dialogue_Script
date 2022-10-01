# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from project_tools import ProjectTools

tools = ProjectTools()
df = tools.clean_data('HIMYM_data_all_characters.csv')
robin_series = tools.get_data_of_characters(df, ['Robin'])
corpus = robin_series.str.cat(sep='\n')
corpus = corpus.replace('\n', ' ')
corpus = corpus.replace('\t', ' ')
for punctuation in ['.', '-', ',', '!', '?', '(', '—', ')']:
    corpus = corpus.replace(punctuation, f'{punctuation} ')

corpus_words = corpus.split(' ')
corpus_words = [word for word in corpus_words if word != '']
corpus = ' '.join(corpus_words)

# create mapping of unique chars to integers
chars = sorted(list(set(corpus)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(corpus)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = corpus[i:i + seq_length]
    seq_out = corpus[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = to_categorical(dataY)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(GRU(1024, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# fit the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)