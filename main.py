import numpy as np
import tensorflow as tf
from tensorflow import keras
from google.colab import drive

drive.mount('/content/gdrive')
train_path = "/content/gdrive/MyDrive/Datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
test_path = "/content/gdrive/MyDrive/Datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
val_path = "/content/gdrive/MyDrive/Datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"

batch_size = 64  
epochs = 100  
latent_dim = 256  

#Preprocessing Data

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(train_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[:len(lines)-1]:
    target_text,input_text, _ = line.split("\t")
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
  