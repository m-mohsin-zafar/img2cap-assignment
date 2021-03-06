import torch
import pandas as pd
import numpy as np

from vocabulary import Vocabulary
from config import *

# Extra Imports
from string import punctuation
from copy import deepcopy


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

    if remove_tags:
        if '<start>' in predicted_caption:
            predicted_caption = predicted_caption.replace('<start>', '')
        if '<end>' in predicted_caption:
            predicted_caption = predicted_caption.replace('<end>', '')
        predicted_caption = predicted_caption.strip()

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


#########################################################################
#
#                   QUESTION 2.2-3 Utility Functions
#
#########################################################################

def captions_to_LoW(cleaned_captions):
    proc_caps = cleaned_captions
    for caption in range(len(proc_caps)):
        proc_caps[caption] = [word for word in proc_caps[caption].split()]
    return proc_caps


def captions_to_LoWIdx(cleaned_captions, vocab):
    proc_caps = cleaned_captions
    for caption in range(len(proc_caps)):
        proc_caps[caption] = [vocab(word) for word in proc_caps[caption].split()]
    return proc_caps


def process_captions(image_ids, cleaned_captions, vocab=None, for_cosine=False):
    if not for_cosine:
        processed_captions = captions_to_LoW(deepcopy(cleaned_captions))
    else:
        processed_captions = captions_to_LoWIdx(deepcopy(cleaned_captions), vocab)

    df = pd.DataFrame([image_ids, processed_captions]).T
    df.columns = ['id', 'captions']
    df = df.groupby(['id'])['captions'].apply(list).reset_index()

    return df


def get_word_embeddings(list_of_caps, model):
    caps = deepcopy(list_of_caps)
    for c in range(len(caps)):
        with torch.no_grad():
            caps[c] = [model.embed(torch.tensor(word_idx)).unsqueeze(0) for word_idx in caps[c]]
    return caps


def get_mean_embedding_vector(vecs):
    return [torch.cat(cap).mean(dim=0).reshape(1, -1).detach().numpy() for cap in vecs]
