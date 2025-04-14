import time
from collections import Counter
import sys

# from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# This is just for measuring training time!
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


"""
def calculate_performance(preds, y):
    
    Returns precision, recall, fscore per batch
  
    # Get the predicted label from the probabilities
    rounded_preds = preds.argmax(1)

    # Calculate the correct predictions batch-wise and calculate precision, recall, and fscore
    # WARNING: Tensors here could be on the GPU, so make sure to copy everything to CPU
    # SET zero_divison to 1 as we may have in our batches instances where not all classes appear
    prec_micro, recall_micro, f1_micro = score(y, preds)
    #precision, recall, fscore, support = precision_recall_fscore_support(
    #    rounded_preds.cpu(), y.cpu(), zero_division=1, average = 'macro'
    #)

    return prec_micro, recall_micro, f1_micro
"""


# Preparing the data loaders for the training and the validation sets
# PyTorch operates on it's own datatype which is very similar to numpy's arrays
# They are called Torch Tensors: https://pytorch.org/docs/stable/tensors.html
# They are optimized for training neural networks
def prepare_dataloader(tr_data, val_data, test_data, word_to_ix, device):
    # First we transform the tweets into one-hot encoded vectors
    # Then we create Torch Tensors from the list of the vectors
    # It is also inportant to send the Tensors to the correct device
    # All of the tensors should be on the same device when training
    tr_data_vecs = torch.FloatTensor(
        word_to_ix.transform(tr_data["text"]).toarray()
    ).to(device)
    tr_labels = torch.LongTensor(tr_data.label.tolist()).to(device)

    val_data_vecs = torch.FloatTensor(
        word_to_ix.transform(val_data["text"]).toarray()
    ).to(device)
    val_labels = torch.LongTensor(val_data.label.tolist()).to(device)

    test_data_vecs = torch.FloatTensor(
        word_to_ix.transform(test_data["text"]).toarray()
    ).to(device)
    test_labels = torch.LongTensor(test_data.label.tolist()).to(device)

    tr_data_loader = [(sample, label) for sample, label in zip(tr_data_vecs, tr_labels)]
    val_data_loader = [
        (sample, label) for sample, label in zip(val_data_vecs, val_labels)
    ]
    test_data_loader = [
        (sample, label) for sample, label in zip(test_data_vecs, test_labels)
    ]

    return tr_data_loader, val_data_loader, test_data_loader


# The DataLoader(https://pytorch.org/docs/stable/data.html) class helps us to prepare the training batches
# It has a lot of useful parameters, one of it is _shuffle_ which will randomize the training dataset in each epoch
# This can also improve the performance of our model
def create_dataloader_iterators(
    tr_data_loader, val_data_loader, test_dataloader, BATCH_SIZE
):
    train_iterator = DataLoader(
        tr_data_loader,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_iterator = DataLoader(
        val_data_loader,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_iterator = DataLoader(
        test_dataloader,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_iterator, valid_iterator, test_iterator


# Prepeare the dataset for the model, by adding the unknown token and the padding token to the vocabulary
def create_input(dataset, analyzer, vocabulary, device):
    dataset_as_indices = []

    # We go through each sentence in the dataset
    # We need to add two additional symbols to the vocabulary, one for the unknown words and one for padding
    for sentence in dataset:
        tokens = analyzer(sentence)
        token_ids = []

        for token in tokens:
            # if the token is in the vocab, we add the id
            if token in vocabulary:
                token_ids.append(vocabulary[token])
            # else we add the id of the unknown token
            else:
                token_ids.append(len(vocabulary))

        # if we removed every token during preprocessing (stopword removal, lemmatization), we add the unknown token to the list so it won't be empty
        if not token_ids:
            token_ids.append(len(vocabulary))
        dataset_as_indices.append(torch.LongTensor(token_ids).to(device))

    return dataset_as_indices


# Preparing the data loaders with padding option for the training and the validation sets
def prepare_dataloader_with_padding(
    tr_data, val_data, test_data, word_to_ix, analyzer, padding_value, device
):
    # First create the id representations of the input vectors
    # Then pad the sequences so all of the input is the same size
    # We padded texts for the whole dataset, this could have been done batch-wise also!
    tr_data_vecs = pad_sequence(
        create_input(
            tr_data["text_with_entity"], analyzer, word_to_ix.vocabulary_, device
        ),
        batch_first=True,
        padding_value=padding_value,
    )
    tr_labels = torch.LongTensor(tr_data["label"].tolist()).to(device)
    tr_lens = torch.LongTensor(
        [
            len(i)
            for i in create_input(
                tr_data["text_with_entity"], analyzer, word_to_ix.vocabulary_, device
            )
        ]  # len of tweets
    )
    # print(tr_lens)
    # print(tr_lens.shape)

    # We also add the texts to the batches
    # This is for the Transformer models, you wont need this in the next experiments
    tr_sents = tr_data["text_with_entity"].tolist()

    val_data_vecs = pad_sequence(
        create_input(
            val_data["text_with_entity"], analyzer, word_to_ix.vocabulary_, device
        ),
        batch_first=True,
        padding_value=padding_value,
    )
    val_labels = torch.LongTensor(val_data["label"].tolist()).to(device)
    val_lens = torch.LongTensor(
        [
            len(i)
            for i in create_input(
                val_data["text_with_entity"], analyzer, word_to_ix.vocabulary_, device
            )
        ]
    )

    val_sents = val_data["text_with_entity"].tolist()

    # Test
    test_data_vecs = pad_sequence(
        create_input(
            test_data["text_with_entity"], analyzer, word_to_ix.vocabulary_, device
        ),
        batch_first=True,
        padding_value=padding_value,
    )
    test_labels = torch.LongTensor(test_data["label"].tolist()).to(device)
    test_lens = torch.LongTensor(
        [
            len(i)
            for i in create_input(
                test_data["text_with_entity"], analyzer, word_to_ix.vocabulary_, device
            )
        ]
    )

    test_sents = test_data["text_with_entity"].tolist()

    tr_data_loader = [
        (sample, label, length, sent)
        for sample, label, length, sent in zip(
            tr_data_vecs, tr_labels, tr_lens, tr_sents
        )
    ]
    val_data_loader = [
        (sample, label, length, sent)
        for sample, label, length, sent in zip(
            val_data_vecs, val_labels, val_lens, val_sents
        )
    ]
    test_data_loader = [
        (sample, label, length, sent)
        for sample, label, length, sent in zip(
            test_data_vecs, test_labels, test_lens, test_sents
        )
    ]

    return tr_data_loader, val_data_loader, test_data_loader


def score(key, prediction, verbose=False):
    """
    Obtain the precision, recall and F1 score from the implementation "Position-aware Attention and Supervised Data Improve Slot Filling".
    INPUT:
        - key: True relation
        - prediction: Relation obtained by the model
    OUTPUT:
        - Returns the metrics specified above
    """
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # NO RELATION is encoded by 0.
    NO_RELATION = 0

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1:
                sys.stdout.write(" ")
            if prec < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1:
                sys.stdout.write(" ")
            if recall < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1:
                sys.stdout.write(" ")
            if f1 < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values())
        )
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values())
        )
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro


def score_str(key, prediction, verbose=False):
    """
    Obtain the precision, recall and F1 score from the implementation "Position-aware Attention and Supervised Data Improve Slot Filling".
    INPUT:
        - key: True relation
        - prediction: Relation obtained by the model
    OUTPUT:
        - Returns the metrics specified above
    """
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # NO RELATION is encoded by 0.
    NO_RELATION = "no_relation"

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1:
                sys.stdout.write(" ")
            if prec < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1:
                sys.stdout.write(" ")
            if recall < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1:
                sys.stdout.write(" ")
            if f1 < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values())
        )
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values())
        )
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro


def extract_predictions(model, test_iterator):
    """
    Takes as input a model and a test data loader, and returns the original labels and the predicted ones.
    """
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for batch in test_iterator:
            text_vecs = batch[0]
            batch_labels = batch[1]
            sen_lens = []
            texts = []

            if len(batch) > 2:
                sen_lens = batch[2]
                texts = batch[3]

            all_labels += batch_labels

            batch_preds = model(text_vecs, sen_lens)
            all_predictions += torch.argmax(batch_preds, dim=1).tolist()
    return all_labels, all_predictions
