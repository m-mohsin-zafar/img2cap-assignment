"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *

# Extra Imports
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# if false, train model; otherwise try loading model from checkpoint and evaluate
EVAL = True
COMPARE_BUT_NOT_EVAL = True

# reconstruct the captions and vocab, just as in extract_features.py
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)
vocab = build_vocab(cleaned_captions)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the models and set the learning parameters
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)

if not EVAL:

    # load the features saved from extract_features.py
    print(len(lines))
    features = torch.load('features.pt', map_location=device)
    print("Loaded features", features.shape)

    features = features.repeat_interleave(5, 0)
    print("Duplicated features", features.shape)

    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,  # change as needed
        shuffle=True,
        num_workers=0,  # may need to set to 0
        collate_fn=caption_collate_fn,  # explicitly overwrite the collate_fn
    )

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print(len(image_ids))
    print(len(cleaned_captions))
    print(features.shape)

    #########################################################################
    #
    #        QUESTION 1.3 Training DecoderRNN
    #
    #########################################################################

    # TODO (Done) write training loop on decoder here

    for epoch in range(NUM_EPOCHS):
        description = f'Training Phase: Epoch {epoch + 1} '
        with tqdm(total=len(dataset_train), desc=description, unit=' img', leave=True) as pbar:
            for batch in train_loader:
                image_features, captions, lengths = batch

                # for each batch, prepare the targets using this torch.nn.utils.rnn function
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # Zero out the gradients for optimizer to avoid accumulation
                optimizer.zero_grad()

                # Models forward pass
                outputs = decoder(image_features, captions, lengths)

                # Calculate Loss
                loss = criterion(outputs, targets)

                # Backpropagation of loss
                loss.backward()

                # Weight Updates
                optimizer.step()

                # Update Visual Progress of tqdm
                pbar.set_postfix(**{'Train CE Loss (running)': loss.item()})
                pbar.update(image_features.shape[0])

    # save model checkpoint after training
    torch.save(decoder, "decoder.ckpt")
    print('Checkpoint Saved!')

# if we already trained, and EVAL == True, reload saved model
elif not COMPARE_BUT_NOT_EVAL:

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),  # using ImageNet norms
                             (0.229, 0.224, 0.225))])

    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)

    # load models
    encoder = EncoderCNN().to(device)
    decoder = torch.load("decoder.ckpt").to(device)
    encoder.eval()
    decoder.eval()  # generate caption, eval mode to not influence batchnorm

    #########################################################################
    #
    #        QUESTION 2.1 Generating predictions on test data
    #
    #########################################################################

    # Reference Vocabulary in accordance with train data
    train_lines = read_lines(TOKEN_FILE_TRAIN)
    train_image_ids, train_cleaned_captions = parse_lines(train_lines)
    ref_vocab = build_vocab(train_cleaned_captions)

    dataset_test = Flickr8k_Images(
        image_ids=test_image_ids[::5],
        transform=data_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    raw_predictions = []
    with tqdm(total=len(dataset_test), desc='Testing Phase', unit=' img', leave=True) as pbar:
        for images in test_loader:
            images.to(device)
            with torch.no_grad():
                # 1. Pass the Images through Encoder to obtain Encoded Representation of Images
                encoded_features = encoder(images)
                encoded_features = encoded_features.flatten(start_dim=1)
                # 2. Sample out the captions from Decoder by passing encoded feature vector
                sampled_ids = decoder.sample(encoded_features)
                raw_predictions.append(sampled_ids)

                pbar.update(images.shape[0])

    raw_predictions = torch.cat(raw_predictions, dim=0)

    predicted_captions = []

    for word_ids in raw_predictions:
        # TODO (Done) define decode_caption() function in utils.py
        decoded_caption = decode_caption(word_ids.numpy(), ref_vocab, remove_tags=True)
        predicted_captions.append(decoded_caption)

    data = {
        'id': dataset_test.image_ids,
        'caption': predicted_captions
    }

    test_predictions_df = pd.DataFrame(data)
    test_predictions_df.to_csv('test_predictions.csv', index=False)
    test_predictions_df.to_json('test_predictions.json', orient='records')

elif COMPARE_BUT_NOT_EVAL:
    #########################################################################
    #
    #        QUESTION 2.2-3 Caption evaluation via text similarity
    #
    #########################################################################

    decoder = torch.load("decoder.ckpt").to(device)
    decoder.eval()  # generate caption, eval mode to not influence batchnorm

    # Read Reference Captions from Test File; and Clean the captions.
    lines = read_lines(TOKEN_FILE_TEST)
    image_ids, cleaned_captions = parse_lines(lines)

    captions_LoW_bleu = process_captions(image_ids, cleaned_captions)
    captions_list_cos = process_captions(image_ids, cleaned_captions, vocab, for_cosine=True)

    bleu_statistics, cos_statistics = [], []

    # Load Predictions from the model
    test_predictions_df = pd.read_json("test_predictions.json")

    # bleu individual n-grams
    n_grams = {
        '1-gram': (1, 0, 0, 0),
        '2-gram': (0, 1, 0, 0),
        '3-gram': (0, 0, 1, 0),
        '4-gram': (0, 0, 0, 1),
        'cumulative': (0.25, 0.25, 0.25, 0.25)
    }

    # For Each Image this Loop will run
    for record in tqdm(test_predictions_df['id']):

        # Filter out reference captions for this 'record' image.
        references_bleu = (captions_LoW_bleu[captions_LoW_bleu['id'] == record]['captions']).to_list()[0]
        references = (captions_list_cos[captions_list_cos['id'] == record]['captions']).to_list()[0]

        # extract candidate caption for this 'record' image.
        predictions = (test_predictions_df[test_predictions_df['id'] == record]['caption']).to_list()[0]
        # for cosine similarity we convert the predicted caption to corresponding list of word ids
        predictions_cos = captions_to_LoWIdx([predictions], vocab)

        # Next we get word embeddings
        references_cos = get_word_embeddings(deepcopy(references), decoder)
        predictions_cos = get_word_embeddings(predictions_cos, decoder)

        # Computing the Mean Embedding Vector for Each
        references_cos = get_mean_embedding_vector(references_cos)
        predictions_cos = get_mean_embedding_vector(predictions_cos)

        # compute cosine similarity score and prepare a list
        cos_score = cosine_similarity(predictions_cos[0], np.array(references_cos).squeeze())  # 2.3
        cos_statistics.append(
            [record, predictions, cos_score.mean()] + [val for pair in zip(references_bleu, cos_score[0]) for val in
                                                       pair])

        # compute BLEU score for reference captions and candidate caption; and prepare a list
        bleu_scores = []
        for w in n_grams.keys():
            bleu_score = sentence_bleu(references_bleu, predictions.split(), weights=n_grams[w])  # 2.2
            bleu_scores.append(bleu_score)

        # bleu scores with smoothing function
        chencherry = SmoothingFunction()
        bleu_scores_smoothed = []
        for w in n_grams.keys():
            bleu_score = sentence_bleu(references_bleu, predictions.split(), weights=n_grams[w],
                                       smoothing_function=chencherry.method1)  # 2.2
            bleu_scores_smoothed.append(bleu_score)
        bleu_statistics.append([record, references_bleu, predictions] + bleu_scores + bleu_scores_smoothed)

    # Export as a DataFrame
    cos_statistics = pd.DataFrame(cos_statistics,
                                  columns=['id', 'prediction', 'mean_cosine_score', 'ref_cap_1', 'cos_scr_1',
                                           'ref_cap_2', 'cos_scr_2', 'ref_cap_3', 'cos_scr_3', 'ref_cap_4', 'cos_scr_4',
                                           'ref_cap_5', 'cos_scr_5'])
    cos_statistics.to_csv("cosine_score_results_final.csv", index=False)

    # Export as a DataFrame
    bleu_cols = [k for k in n_grams.keys()]
    bleu_cols_smoothed = [k + '_smoothed' for k in n_grams.keys()]
    bleu_statistics = pd.DataFrame(bleu_statistics,
                                   columns=['id', 'reference', 'predicted'] + bleu_cols + bleu_cols_smoothed)
    bleu_statistics.to_csv("bleu_score_results.csv", index=False)
