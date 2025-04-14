import os
import random
import torch
from load_data.loader import DataLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab


def evaluate_saved_model(
    model_dir="./saved_models/00",
    model="/best_model.pt",
    data_dir="../../../data/tacred/json",
    dataset="test",
    out="",
    seed=1234,
    cuda=torch.cuda.is_available(),
    cpu=False,
):
    torch.manual_seed(seed)
    random.seed(1234)
    if cpu:
        cuda = False
    elif cuda:
        torch.cuda.manual_seed(seed)

    # Load opt
    model_file = model_dir + model
    print("Loading model from {}".format(model_file))
    opt = torch_utils.load_config(model_file)
    model = RelationModel(opt)
    model.load(model_file)

    # Load vocab
    vocab_file = "./vocab.pkl"
    vocab = Vocab(vocab_file, load=True)
    assert (
        opt["vocab_size"] == vocab.size
    ), "Vocab size must match that in the saved model."

    # Load data
    data_file = data_dir + "/" + dataset + ".json"
    print(
        "Loading data from {} with batch size {}...".format(
            data_file, opt["batch_size"]
        )
    )
    batch = DataLoader(data_file, opt["batch_size"], opt, vocab, evaluation=True)

    helper.print_config(opt)
    id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])

    predictions = []
    all_probs = []
    for i, b in enumerate(batch):
        preds, probs, _ = model.predict(b)
        predictions += preds
        all_probs += probs
    predictions = [id2label[p] for p in predictions]
    p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

    # Writing gold labels and prediction to a text file in column format
    if len(out) > 0:
        helper.ensure_dir(os.path.dirname(out))
        with open(out, "w") as file:
            file.write("Gold Label\tPrediction\n")
            for pred, gold_label in zip(predictions, batch.gold()):
                file.write(f"{gold_label}\t{pred}\n")
        print("Predictions saved to {}.".format(out))

    print("Evaluation ended.")
