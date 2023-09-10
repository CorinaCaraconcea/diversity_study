import os
import json
import numpy
import re
import torch
import torch_ac
import gymnasium as gym

import rl_starter_files.utils


def get_obss_preprocessor(obs_space):
    # the function checks if the observed space is an image space
    # then it processes the observation and wraps it around a DictList

    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })

        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))
# The returned obs_space is a dictionary that describes the observation space 
# (which might be different from the input obs_space), and preprocess_obss 
# is the preprocessing function that's been defined.
    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

# initializes a max size for the vocabulary and an empty vocabulary
    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

# load an existing vocabulary into the class
    def load_vocab(self, vocab):
        self.vocab = vocab

# check if the token is in the vocabulary, if it is return its id
# if the vocab has reached the max size, then return an error
# if the token is not in the vocab add it and return a new id
    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
