"""
Prepare Pan Tadeusz dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

DICTIONARY = ['\n', ' ', '!', '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '«', '»', 'Ó', 'à', 'ó', 'ą', 'Ć', 'ć', 'ę', 'Ł', 'ł', 'ń', 'Ś', 'ś', 'Ź', 'ź', 'Ż', 'ż', '—', '…']


with open("raw/pan-tadeusz.txt", 'r') as f:
    data = f.read()

data = "".join([char for char in data if char in DICTIONARY])

print(f"length of dataset in characters: {len(data):,}")

print("all the unique characters:", ''.join(DICTIONARY))
print(f"vocab size: {len(DICTIONARY):,}")

stoi = { ch:i for i,ch in enumerate(DICTIONARY) }
itos = { i:ch for i,ch in enumerate(DICTIONARY) }

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': len(DICTIONARY),
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters: 445,655
# all the unique characters:
#  !()*,-./0123456789:;?ABCDEFGHIJKLMNOPRSTUVWZabcdefghijklmnopqrstuvwxyz«»ÓàóąĆćęŁłńŚśŹźŻż—…
# vocab size: 92
# train has 401,089 tokens
# val has 44,566 tokens
