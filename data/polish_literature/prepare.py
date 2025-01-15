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
from itertools import chain

DICTIONARY = ['\n', ' ', '!', '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '«', '»', 'Ó', 'à', 'ó', 'ą', 'Ć', 'ć', 'ę', 'Ł', 'ł', 'ń', 'Ś', 'ś', 'Ź', 'ź', 'Ż', 'ż', '—', '…']

STOI = { ch:i for i,ch in enumerate(DICTIONARY) }
ITOS = { i:ch for i,ch in enumerate(DICTIONARY) }

def encode(s: str) -> list[int]:
    return [STOI[c] for c in s] # encoder: take a string, output a list of integers
def decode(l: str) -> list[int]:
    return ''.join([ITOS[i] for i in l]) # decoder: take a list of integers, output a string

def process_source(file_name: str) -> tuple[list[int], list[int]]:
    with open(f"raw/{file_name}", 'r') as f:
        data = f.read()

    # Filter only characters that are in the dictionary
    data = "".join([char for char in data if char in DICTIONARY])

    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    train_ids = encode(train_data)
    val_ids = encode(val_data)

    return train_ids, val_ids

sources = [
    "balladyna.txt",
    "dziady.txt",
    "lalka.txt",
    "quo-vadis.txt",
    "w-pustyni-i-w-puszczy.txt",
    "zemsta.txt",
]

train_sources = []
val_sources = []

for file in sources:
    train_ids, val_ids = process_source(file)
    train_sources.append(train_ids)
    val_sources.append(val_ids)

train_data = list(chain.from_iterable(train_sources))
val_data = list(chain.from_iterable(val_sources))

print(f"length of dataset in characters: {len(train_data) + len(val_data):,}")

print("all the unique characters:", ''.join(DICTIONARY))
print(f"vocab size: {len(DICTIONARY):,}")

print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")

# export to bin files
train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': len(DICTIONARY),
    'itos': ITOS,
    'stoi': STOI,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
