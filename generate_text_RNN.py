from project_tools import ProjectTools
import tensorflow as tf

tools = ProjectTools()
df = tools.clean_data('HIMYM_data_all_characters.csv')
barney_df = tools.get_data_of_characters(df, ['Barney'])
robin_df = tools.get_data_of_characters(df, ['Robin'])
WEIGHTS_PATH_BARNEY = './training_checkpoints/ckpt_Barney_gru_model'
WEIGHTS_PATH_ROBIN = './training_checkpoints/ckpt_Robin_gru_model'
TEMPERATURE = 0.8


class GRUModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
          states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


def get_ids_chars_from_lookup(df):
    corpus = df.str.cat(sep='\n')
    corpus = corpus.replace('\n', ' ')
    corpus = corpus.replace('\t', ' ')
    for punctuation in ['.', '-', ',', '!', '?', '(', '—', ')']:
        corpus = corpus.replace(punctuation, f'{punctuation} ')

    corpus_words = corpus.split(' ')
    corpus_words = [word for word in corpus_words if word != '']
    corpus = ' '.join(corpus_words)
    vocab = sorted(set(corpus))
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(),
                                                  invert=True, mask_token=None)
    return ids_from_chars, chars_from_ids


def generate_text(seed, _model, _chars_from_ids, _ids_from_chars):
    one_step_model = OneStep(_model, _chars_from_ids, _ids_from_chars, TEMPERATURE)
    states = None
    next_char = tf.constant([seed])
    result = [next_char]

    for n in range(70):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result[:-1])
    return result[0].numpy().decode('utf-8')


barney_ids_from_chars, barney_chars_from_ids = get_ids_chars_from_lookup(barney_df)
robin_ids_from_chars, robin_chars_from_ids = get_ids_chars_from_lookup(robin_df)

barney_model = GRUModel(vocab_size=len(barney_ids_from_chars.get_vocabulary()), embedding_dim=512, rnn_units=1024)
barney_model.load_weights(WEIGHTS_PATH_BARNEY)
robin_model = GRUModel(vocab_size=len(robin_ids_from_chars.get_vocabulary()), embedding_dim=512, rnn_units=1024)
robin_model.load_weights(WEIGHTS_PATH_ROBIN)


def slice_string_at_sentence_ending_punctuation(text: str) -> str:
    period_index = text.rfind('.')
    question_mark_index = text.rfind('?')
    if period_index > 0 or question_mark_index > 0:
        punct_idx = max(period_index, question_mark_index)
        return text[:punct_idx+1]
    # edge case - no sentence ending punctuation
    return text


def delete_text_gen_noise(text: str) -> str:
    return text.replace(' . . ', ' ').replace(' . ', '')


original_seed = 'Are you Canadian?'
barney_history = ''
robin_history = ''

barney_response = generate_text(original_seed, barney_model, barney_chars_from_ids, barney_ids_from_chars)
barney_response = barney_response[len(original_seed)+1:]
barney_response = slice_string_at_sentence_ending_punctuation(barney_response)
barney_response = delete_text_gen_noise(barney_response)
barney_history += barney_response
print(f"Seed: {original_seed}")
print(f"Barney: {barney_response}")

for step in range(5):
    robin_response = generate_text(barney_history, robin_model, robin_chars_from_ids, robin_ids_from_chars)
    robin_response = ' '.join(robin_response.split()[len(barney_history.split()):]) \
        if len(robin_history) > 0 else robin_response[len(original_seed)+1:]
    robin_response = slice_string_at_sentence_ending_punctuation(robin_response)
    robin_response = delete_text_gen_noise(robin_response)
    print(f'Robin: {robin_response}')
    robin_history += f' {robin_response}'

    barney_response = generate_text(robin_history, barney_model, barney_chars_from_ids, barney_ids_from_chars)
    barney_response = ' '.join(barney_response.split()[len(robin_history.split()):])
    barney_response = slice_string_at_sentence_ending_punctuation(barney_response)
    barney_response = delete_text_gen_noise(barney_response)
    print(f'Barney: {barney_response}')
    barney_history += f' {barney_response}'

