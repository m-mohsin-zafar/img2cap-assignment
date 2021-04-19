import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

from vocabulary import Vocabulary
from config import *

# Extra Imports
from string import punctuation


def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True:
        # Get next line from file 
        line = file.readline()
        if not line:
            break
        lines.append(line.strip())
    file.close()
    return lines


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    image_ids = []
    cleaned_captions = []

    # QUESTION 1.1

    for line in lines:
        # first we split the image id from caption text based on \t
        id = line.split('\t')[0]
        # then we extract remove .jpg#x part from image id (where x = 1 to 5)
        id = id.split('.')[0]
        # finally we extract raw text caption
        raw_caption = line.split('\t')[1]
        # and forward to other function for cleaning the text
        caption = clean_caption(raw_caption)

        image_ids.append(id)
        cleaned_captions.append(caption)

    return image_ids, cleaned_captions


def clean_caption(raw_caption):
    # convert to lower case
    caption = raw_caption.lower()
    # remove punctuations / special characters
    caption = ''.join(c for c in caption if c not in punctuation)
    # remove digits
    caption = ''.join(c for c in caption if not c.isdigit())
    # remove extra spaces on left or right side of the string
    caption = caption.strip()

    return caption


def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with

    Returns:
        vocab (Vocabulary): Vocabulary object
    """
    # QUESTION 1.1
    # Here we Build a vocabulary

    # create a vocab instance
    vocab = Vocabulary()

    words = dict()
    for caption in cleaned_captions:  # iterate through all cleaned_caption
        for word in caption.split():  # iterate over all words in a caption
            # add the token words to vocabulary if and only if the count of word is more than 3
            if word not in words.keys():
                words[word] = 1
            else:
                words[word] += 1
                if words[word] > 3:
                    vocab.add_word(word)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    print(vocab.idx)

    return vocab


def decode_caption(sampled_ids, vocab):
    """ 
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """
    predicted_caption = ""

    # QUESTION 2.1

    return predicted_caption


"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""


def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0)

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths
