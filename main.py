import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.utils.vis_utils import plot_model

from google.colab import drive

drive.mount('/content/gdrive')
train_path = "/content/gdrive/MyDrive/Datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
test_path = "/content/gdrive/MyDrive/Datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
val_path = "/content/gdrive/MyDrive/Datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"



is_embedding_used = False
def read_data(data_path):
  input_texts = []
  target_texts = []
  input_characters = set()
  target_characters = set()

  with open(data_path, "r", encoding="utf-8") as f:
      lines = f.read().split("\n")
  num_of_samples = len(lines)-1
  for line in lines[:num_of_samples]:
      target_text,input_text, _ = line.split("\t")
      # We use "tab" as the "start sequence" character
      # for the targets, and "\n" as "end sequence" character.
      target_text = "\t" + target_text + "\n"
      input_texts.append(input_text)
      target_texts.append(target_text)

      for char in input_text:
          if char not in input_characters:
              input_characters.add(char)
      for char in target_text:
          if char not in target_characters:
              target_characters.add(char)
  return input_texts, target_texts , input_characters, target_characters

def input_data(input_texts, target_texts, max_encoder_seq_length, num_encoder_tokens, max_decoder_seq_length, num_decoder_tokens,input_token_index,target_token_index ):
  if is_embedding_used:
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="float32")
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype="float32")
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
      for t, char in enumerate(input_text):
          encoder_input_data[i, t] = input_token_index[char]
      encoder_input_data[i, t + 1 :] = input_token_index[" "]
      for t, char in enumerate(target_text):
          # decoder_target_data is ahead of decoder_input_data by one timestep
          decoder_input_data[i, t] =  target_token_index[char]
          if t > 0:
              # decoder_target_data will be ahead by one timestep
              # and will not include the start character.
              decoder_target_data[i, t - 1, target_token_index[char]] =  1
      decoder_input_data[i, t + 1 :] = target_token_index[" "]
      decoder_target_data[i, t:] = target_token_index[" "]
  else:
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
      for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
      encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
      for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
          decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
      decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
      decoder_target_data[i, t:, target_token_index[" "]] = 1.0
  return encoder_input_data, decoder_input_data, decoder_target_data

def get_info(input_texts, target_texts , input_characters, target_characters):
  input_characters = sorted(list(input_characters))
  input_characters.append(" ")
  target_characters = sorted(list(target_characters))
  target_characters.append(" ")
  num_encoder_tokens = len(input_characters)
  num_decoder_tokens = len(target_characters)
  max_encoder_seq_length = max([len(txt) for txt in input_texts])+1
  max_decoder_seq_length = max([len(txt) for txt in target_texts])
  input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
  target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
  return num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index


input_texts, target_texts , input_characters, target_characters = read_data(train_path)
num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index = get_info(input_texts, target_texts , input_characters, target_characters)
encoder_input_train_data, decoder_input_train_data, decoder_target_train_data = input_data(input_texts, target_texts, max_encoder_seq_length, num_encoder_tokens, max_decoder_seq_length, num_decoder_tokens,input_token_index,target_token_index)

input_texts, target_texts , _, _ = read_data(val_path)
encoder_input_val_data, decoder_input_val_data, decoder_target_val_data = input_data(input_texts, target_texts, max_encoder_seq_length, num_encoder_tokens, max_decoder_seq_length, num_decoder_tokens,input_token_index,target_token_index)


#-------

encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

enc_embedding_dims = 64
dec_embedding_dims = 128
if is_embedding_used:
  encoder_inputs = keras.Input(shape=(max_encoder_seq_length))
  decoder_inputs = keras.Input(shape=(max_decoder_seq_length))  
  embedded_encoder_inputs = keras.layers.Embedding(input_dim = num_encoder_tokens, output_dim = enc_embedding_dims, input_length = max_encoder_seq_length, name = "enc_embedding")(encoder_inputs)
  embedded_decoder_inputs = keras.layers.Embedding(input_dim= num_decoder_tokens, output_dim = dec_embedding_dims, input_length= max_decoder_seq_length,  name = "dec_embedding")(decoder_inputs)
else:
  embedded_encoder_inputs = encoder_inputs
  embedded_decoder_inputs = decoder_inputs
def build_encoder(inputs, cell_type, latent_dims =[256], embedding_dims = 256):

  current_state = None
  #return_state = False
  is_initial_state_recurrent = False  #works when all cells in encoder are of same type or combination of GRU and RNN
  dropout = 0.2
  recurrent_dropout = 0
  for i, latent_dim in enumerate(latent_dims):
    if cell_type[i] == "RNN":
      cell = keras.layers.SimpleRNN(latent_dim, return_sequences=True, return_state=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
      whole_sequence_output, final_state = cell(inputs, initial_state = current_state)
      inputs = whole_sequence_output
      current_state = final_state
    elif cell_type[i] == "LSTM":
      cell = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
      whole_sequence_output, state_h, state_c = cell(inputs, initial_state = current_state)
      inputs = whole_sequence_output
      current_state = [state_h, state_c]
    elif cell_type[i] == "GRU":
      cell = keras.layers.GRU(latent_dim, return_sequences=True, return_state=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
      whole_sequence_output, final_state = cell(inputs, initial_state = current_state)
      inputs = whole_sequence_output
      current_state = final_state

    else:
      raise Exception("Invalid cell")

   
    '''
    #simple_rnn = keras.layers.SimpleRNN(latent_dim,return_sequences=True, return_state=True)
    if cell_type == "LSTM":
     
      #initial_state = [state_h, state_c]
    else:
      whole_sequence_output, final_state = cell(inputs, initial_state=initial_state)
      inputs = whole_sequence_output
      #initial_state = final_state
    '''
  if cell_type[i] == "LSTM":
    encoder_state = [state_h, state_c]
  else:
    encoder_state = final_state
 
  return whole_sequence_output, encoder_state


def build_decoder(inputs, initial_state, cell_type, latent_dims =[256], embedding_dims = 256):
  current_state = initial_state
  return_sequences = True
  is_initial_state_recurrent = False  #works when all cells in encoder are of same type or combination of GRU and RNN
  is_initial_state_encoders = True
  dropout = 0.2
  recurrent_dropout = 0
  for i, latent_dim in enumerate(latent_dims):
    #if i == (len(latent_dims)-1):
      #return_sequences = False
    if cell_type[i] == "RNN":
      cell = keras.layers.SimpleRNN(latent_dim,return_sequences=return_sequences, return_state=True, dropout=dropout, recurrent_dropout=recurrent_dropout, name = "decoder_"+str(i))
    elif cell_type[i] == "LSTM":
      cell = keras.layers.LSTM(latent_dim, return_sequences=return_sequences, return_state=True, dropout=dropout, recurrent_dropout=recurrent_dropout, name = "decoder_"+str(i))
    elif cell_type[i] == "GRU":
      cell = keras.layers.GRU(latent_dim, return_sequences=return_sequences, return_state=True, dropout=dropout, recurrent_dropout=recurrent_dropout, name = "decoder_"+str(i))
    else:
      raise Exception("Invalid cell")

    if cell_type[i] == "LSTM":
      whole_sequence_output,state_h, state_c = cell(inputs, initial_state=current_state)
      inputs = whole_sequence_output
      current_state = [state_h, state_c]
    else:
      whole_sequence_output, final_state = cell(inputs, initial_state=current_state)
      inputs = whole_sequence_output
      current_state = final_state

    if not is_initial_state_recurrent:
      current_state = None
    if is_initial_state_encoders:
      current_state = initial_state
    #whole_sequence_output, final_state = simple_rnn(inputs, initial_state=initial_state)
    #inputs = whole_sequence_output
    #initial_state = final_state

  return whole_sequence_output

enc_depth = 1
dec_depth = 1
batch_size = 64 
epochs = 30
is_attention_used = False
enc_latent_dims = [256]*enc_depth 
enc_cell_type = ["LSTM"]*enc_depth
#enc_embedding_dims = 256

dec_latent_dims = [256]*dec_depth 
dec_cell_type = ["LSTM"]*dec_depth
#dec_embedding_dims = 256
  


enc_outputs, enc_states =  build_encoder(embedded_encoder_inputs,cell_type = enc_cell_type,latent_dims = enc_latent_dims, embedding_dims = enc_embedding_dims )
dec_outputs = build_decoder(embedded_decoder_inputs , enc_states, cell_type = dec_cell_type,latent_dims = dec_latent_dims, embedding_dims = dec_embedding_dims )

if is_attention_used:
  attn_layer = keras.layers.Attention(name='attention_layer')
  attn_out = attn_layer([enc_outputs, dec_outputs])
  decoder_concat_input = keras.layers.Concatenate(axis=-1, name='concat_layer')([dec_outputs, attn_out])
  dense = keras.layers.Dense(num_decoder_tokens, activation='softmax', name='softmax_layer')
  dense_time = dense#(dense, name='time_distributed_layer')
  dec_outputs = dense_time(decoder_concat_input)
else:
  dec_dense = (keras.layers.Dense(num_decoder_tokens, activation="softmax", name = "dense_0"))
  dec_outputs = dec_dense(dec_outputs)

model = keras.Model([encoder_inputs, decoder_inputs], dec_outputs)

callback = keras.callbacks.EarlyStopping(monitor="loss", patience=4)
model.compile(
    optimizer= keras.optimizers.Adam(learning_rate=0.01), loss=keras.losses.categorical_crossentropy, metrics="accuracy"
)
model.fit(
    [encoder_input_train_data, decoder_input_train_data],
    decoder_target_train_data,
    batch_size=batch_size,
    epochs=epochs,
    callbacks = [callback],
    #validation_split = .2
    validation_data = ([encoder_input_val_data, decoder_input_val_data],decoder_target_val_data),
)
# Save model
#model.save("s2s")