import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from random import random
import re


class StochasticTextGenerator:
    def __init__(self, data_set):
        """
        :param data_set: A Pandas series with text from a specific character/person
        """
        self.data_set = data_set
        if not isinstance(data_set, pd.Series):
            raise TypeError('Input data is not supported. It has to be a Pandas Series type')

        # preprocess raw data
        self.corpus = self.data_set.str.cat(sep='\n')
        self.corpus = self.corpus.replace('\n', ' ')
        self.corpus = self.corpus.replace('\t', ' ')
        for punctuation in ['.', '-', ',', '!', '?', '(', '—', ')']:
            self.corpus = self.corpus.replace(punctuation, f' {punctuation} ')

        self.corpus_words = self.corpus.split(' ')
        self.corpus_words = [word for word in self.corpus_words if word != '']

        self.distinct_words = list(set(self.corpus_words))
        self.word_idx_dict = {word: i for i, word in enumerate(self.distinct_words)}
        self.distinct_words_count = len(list(set(self.corpus_words)))
        self.next_word_matrix = np.zeros([self.distinct_words_count, self.distinct_words_count])

        for i, word in enumerate(self.corpus_words[:-1]):
            first_word_idx = self.word_idx_dict[word]
            next_word_idx = self.word_idx_dict[self.corpus_words[i + 1]]
            self.next_word_matrix[first_word_idx][next_word_idx] += 1

    """
    Naive chain method.
    
    Our baseline method - plug in the most likely word after each word and create a text this way. It manages to 
    generate text up to a certain point, which after it completes the rest of the length specified with dots. Also,
    it's naive in the way that it can only take one word as a seed.
    """

    def _most_likely_word_after(self, word):
        most_likely = self.next_word_matrix[self.word_idx_dict[word]].argmax()
        return self.distinct_words[most_likely]

    def naive_chain(self, seed, length=15):
        """
        :param seed: The input sequence to sample from and generate text
        :param length: Amount of words expected as a seed to generate from
        :return: A generated text where each word is the word with the highest probability to come after
        the previous one
        """
        current_word = seed
        sentence = seed
        for _ in range(length):
            sentence += ' '
            next_word = self._most_likely_word_after(current_word)
            sentence += next_word
            current_word = next_word
        return sentence

    """
    Markov chain method.
    
    What are Markov Chains?

    A Markov Chain is a stochastic process that models a finite set of states, with fixed conditional probabilities of 
    jumping from a given state to another.
    
    What this means is, we will have an “agent” that randomly jumps around different states, with a certain probability 
    of going from each state to another one.
    
    To utilize this in text generation, we will start with a seed. After that, we will use the previous k words as the 
    current state, and model the probabilities of the next token. 
    
    So, at each iteration we will have a new seed that will be considered at generating the next word. 
    We will only have to specify the first seed.
    """

    @staticmethod
    def _weighted_choice(objects, weights):
        """
        returns randomly an element from the sequence of 'objects', the likelihood of the objects is weighted
        according to the sequence of 'weights', i.e. percentages.
        """
        weights = np.array(weights, dtype=np.float64)
        sum_of_weights = weights.sum()

        # standardization:
        np.multiply(weights, 1 / sum_of_weights, weights)
        weights = weights.cumsum()

        x = random()
        for i in range(len(weights)):
            if x < weights[i]:
                return objects[i]

    def _sample_next_word_after_sequence(self, word_sequence, seed_length, alpha=0):
        """
        :param word_sequence: The input sequence to sample from and generate text
        :param seed_length: Amount of words expected as a seed to generate from, at each iteration (k)
        :param alpha: The chance that a totally random word instead of the ones suggested by the corpus will be picked.
        :return: The next word calculated through a weighted choice
        """
        sets_of_k_words = [' '.join(self.corpus_words[i:i + seed_length])
                           for i, _ in enumerate(self.corpus_words[:-seed_length])]

        sets_count = len(list(set(sets_of_k_words)))
        next_after_k_words_matrix = dok_matrix((sets_count, len(self.distinct_words)))

        distinct_sets_of_k_words = list(set(sets_of_k_words))
        k_words_idx_dict = {word: i for i, word in enumerate(distinct_sets_of_k_words)}

        for i, word in enumerate(sets_of_k_words[:-seed_length]):
            word_sequence_idx = k_words_idx_dict[word]
            next_word_idx = self.word_idx_dict[self.corpus_words[i + seed_length]]
            next_after_k_words_matrix[word_sequence_idx, next_word_idx] += 1

        try:
            next_word_vector = next_after_k_words_matrix[k_words_idx_dict[word_sequence]] + alpha
            likelihoods = next_word_vector / next_word_vector.sum()
        except KeyError as ke:
            print(f"{ke} is an invalid seed since it's not in the corpus")
            raise

        return self._weighted_choice(self.distinct_words, likelihoods.toarray())

    def markov_chain(self, seed, seed_length, chain_length=15):
        """
        :param seed: The input sequence to sample from and generate text at each iteration.
        :param seed_length: Amount of words expected as a seed to generate from.
        :param chain_length: Length of text generated (in words)
        :return: A generated text where each word is a "weighted choice", while considering at each iteration a
        "new" seed of k words. The generated text is cleaned a bit before the function returns.
        """
        current_words = seed.split(' ')
        if len(current_words) != seed_length:
            raise ValueError(f'wrong number of words, expected {seed_length}')
        sentence = seed

        for _ in range(chain_length):
            sentence += ' '
            next_word = self._sample_next_word_after_sequence(' '.join(current_words), seed_length)
            sentence += next_word
            current_words = current_words[1:] + [next_word]
        return re.sub(r'\s([?.!",](?:\s|$))', r'\1', sentence)
