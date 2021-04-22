import torch
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

from nltk.translate.bleu_score import sentence_bleu

from vocabulary import Vocabulary
from config import *

# Extra Imports
from string import punctuation

def listed_captions(cleaned_captions, vocab):
    for caption in range(len(cleaned_captions)):
        caption_encoded = []
        for word in cleaned_captions[caption].split():
            caption_encoded.append((word))
        cleaned_captions[caption] = caption_encoded
    return cleaned_captions

def combine_im_captions(image_ids, cleaned_captions, vocab):
    cleaned_caption = listed_captions(cleaned_captions, vocab)
    two_cols = pd.DataFrame([image_ids, cleaned_caption]).T
    two_cols.columns = ['image_ids', 'captions']
    two_cols = two_cols.groupby(['image_ids'])['captions'].apply(list).reset_index()
    return two_cols

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
            # add the token words to vocabulary if and only if the count of word is more than MIN_FREQUENCY i.e. 3
            if word not in words.keys():
                words[word] = 1
            else:
                words[word] += 1
                if words[word] > MIN_FREQUENCY:
                    vocab.add_word(word)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    print(vocab.idx)

    return vocab


def decode_caption(sampled_ids, vocab, remove_tags=False):
    """ 
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """

    # QUESTION 2.1

    predicted_caption = ' '.join(vocab.idx2word[id] for id in sampled_ids)
    index_for_end = predicted_caption.find('<end>')
    predicted_caption = predicted_caption[: index_for_end+5]

    if remove_tags:
        predicted_caption = predicted_caption[7:index_for_end].strip()

    return predicted_caption


"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""

def CosineSimilarity(references, caption, vocab):
    score = []
    caption_vector = [1 if w in caption else 0 for w in list(vocab.word2idx.keys())]
    for reference in references:
        reference_vector = [1 if w in reference else 0 for w in list(vocab.word2idx.keys())]
        score.append(cosine(reference_vector, caption_vector))
    return np.mean(score)

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
