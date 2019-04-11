from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import random
from utils import *



class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    
    self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                     return_sequences=True, 
                                     return_state=True, 
                                     recurrent_initializer='glorot_uniform')


  def call(self, x, hidden):
    x = self.embedding(x)
    output, state_h,state_c = self.lstm(x, initial_state=hidden)        
    return output, state_h, state_c 

  def initialize_hidden_state(self):
    return [tf.zeros((self.batch_sz, self.enc_units)),tf.zeros((self.batch_sz, self.enc_units))]

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.dec_units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

  def call(self, x, hidden, enc_output):     

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # passing the concatenated vector to the GRU
    output, h_state, c_state = self.lstm(x, initial_state=hidden)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, h_state , c_state

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
  
    return tf.reduce_mean(loss_)

def evaluate(sentence):    
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], 
                                                           maxlen=max_length_inp, 
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_out, enc_hidden_h, enc_hidden_c  = encoder(inputs, hidden)

    dec_hidden = [enc_hidden_h, enc_hidden_c]
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    for t in range(max_length_targ):
        predictions, dec_hidden_h, dec_hidden_c = decoder(dec_input, 
                                                             dec_hidden, 
                                                             enc_out)
        dec_hidden = [dec_hidden_h, dec_hidden_c]
        
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

def translate(sentence):
    result, sentence = evaluate(sentence)        
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

@tf.function
def train_step(inp, targ, enc_hidden, teacher_forcing_ratio = 0.5):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden_h, enc_hidden_c = encoder(inp, enc_hidden)

        dec_hidden = [enc_hidden_h, enc_hidden_c]

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)       

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            teacher_force = random.random() < teacher_forcing_ratio

            # passing enc_output to the decoder
            predictions, dec_hidden_h, dec_hidden_c = decoder(dec_input, dec_hidden, enc_output)
            dec_hidden = [dec_hidden_h, dec_hidden_c]

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            predict_ids = tf.expand_dims(tf.argmax(predictions, axis=1), 1)

            dec_input = tf.expand_dims(targ[:, t], 1) if teacher_force else predict_ids

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss

if __name__ == "__main__":
    # Try experimenting with the size of that dataset
    num_examples = 30000

    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset("http://www.manythings.org/anki/fra-eng.zip","fra-eng.zip", num_examples)

    # Calculate max_length of the target tensors
    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    
    EPOCHS = 30

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    translate(u"c'est ma vie")

    translate(u"je suis a la maison")

    translate(u"il fait tres froid")    
