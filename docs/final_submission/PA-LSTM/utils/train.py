import os
from datetime import datetime
import time
import numpy as np
import random
import torch
from shutil import copyfile
from utils import scorer, constant, helper
from utils.vocab import Vocab
from load_data.loader import DataLoader
from model.rnn import RelationModel


def model_train(
    data_dir="../../data/",
    vocab_dir="../../docs/final_submission/PA-LSTM",
    emb_dim=300,
    ner_dim=30,
    pos_dim=30,
    hidden_dim=200,
    num_layers=2,
    dropout=0.5,
    word_dropout=0.04,
    topn=1e10,
    lower=False,
    attn=True,
    attn_dim=200,
    pe_dim=30,
    lr=1.0,
    optim="sgd",
    num_epoch=30,
    batch_size=50,
    max_grad_norm=5.0,
    log_step=20,
    log="logs.txt",
    save_epoch=15,
    save_dir="./saved_models",
    id="00",
    info="",
    seed=1234,
    cuda=torch.cuda.is_available(),
    cpu=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(1234)
    if cpu:
        cuda = False
    elif cuda:
        torch.cuda.manual_seed(seed)

    # Make opt
    opt = {
        "data_dir": data_dir,
        "vocab_dir": vocab_dir,
        "emb_dim": emb_dim,
        "ner_dim": ner_dim,
        "pos_dim": pos_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "word_dropout": word_dropout,
        "topn": topn,
        "lower": lower,
        "attn": attn,
        "attn_dim": attn_dim,
        "pe_dim": pe_dim,
        "lr": lr,
        "optim": optim,
        "num_epoch": num_epoch,
        "batch_size": batch_size,
        "max_grad_norm": max_grad_norm,
        "log_step": log_step,
        "log": log,
        "save_epoch": save_epoch,
        "save_dir": save_dir,
        "id": id,
        "info": info,
        "seed": seed,
        "cuda": cuda,
        "cpu": cpu,
    }
    opt["num_class"] = len(constant.LABEL_TO_ID)

    # Load vocab
    vocab_file = vocab_dir + "/vocab.pkl"
    vocab = Vocab(vocab_file, load=True)
    opt["vocab_size"] = vocab.size
    emb_file = vocab_dir + "/embedding.npy"
    emb_matrix = np.load(emb_file)
    assert emb_matrix.shape[0] == vocab.size
    assert emb_matrix.shape[1] == opt["emb_dim"]

    # Load data
    print(
        "Loading data from {} with batch size {}...".format(
            opt["data_dir"], opt["batch_size"]
        )
    )
    train_batch = DataLoader(
        data_dir + "tacred/json/train.json",
        opt["batch_size"],
        opt,
        vocab,
        evaluation=False,
    )
    tacred_dev_batch = DataLoader(
        data_dir + "tacred/json/dev.json",
        opt["batch_size"],
        opt,
        vocab,
        evaluation=True,
    )
    tacrev_dev_batch = DataLoader(
        data_dir + "tacrev/json/dev.json",
        opt["batch_size"],
        opt,
        vocab,
        evaluation=True,
    )

    model_id = opt["id"] if len(opt["id"]) > 1 else "0" + opt["id"]
    model_save_dir = save_dir + "/" + model_id
    opt["model_save_dir"] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)

    # Save config
    helper.save_config(opt, model_save_dir + "/config.json", verbose=True)
    vocab.save("./vocab.pkl")
    file_logger = helper.FileLogger(
        model_save_dir + "/" + opt["log"],
        header="# epoch\ttrain_loss\ttacred_dev_loss\ttacred_dev_f1\ttacrev_dev_loss\ttacrev_dev_f1",
    )

    # Print model info
    helper.print_config(opt)

    # Model
    model = RelationModel(opt, emb_matrix=emb_matrix)

    id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
    dev_f1_history_tacred = []
    dev_f1_history_tacrev = []
    current_lr = opt["lr"]

    global_step = 0
    format_str = (
        "{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}"
    )
    max_steps = len(train_batch) * opt["num_epoch"]

    # Start training
    for epoch in range(1, opt["num_epoch"] + 1):
        train_loss = 0
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss = model.update(batch)
            train_loss += loss
            if global_step % opt["log_step"] == 0:
                duration = time.time() - start_time
                print(
                    format_str.format(
                        datetime.now(),
                        global_step,
                        max_steps,
                        epoch,
                        opt["num_epoch"],
                        loss,
                        duration,
                        current_lr,
                    )
                )

        # Evaluate on tacred dev
        print("Evaluating on TACRED dev set...")
        predictions_tacred = []
        dev_loss_tacred = 0
        for i, batch in enumerate(tacred_dev_batch):
            preds, _, loss = model.predict(batch)
            predictions_tacred += preds
            dev_loss_tacred += loss
        predictions_tacred = [id2label[p] for p in predictions_tacred]
        _, _, dev_f1_tacred = scorer.score(tacred_dev_batch.gold(), predictions_tacred)

        # Evaluate on tacrev dev
        print("Evaluating on TACREV dev set...")
        predictions_tacrev = []
        dev_loss_tacrev = 0
        for i, batch in enumerate(tacrev_dev_batch):
            preds, _, loss = model.predict(batch)
            predictions_tacrev += preds
            dev_loss_tacrev += loss
        predictions_tacrev = [id2label[p] for p in predictions_tacrev]
        _, _, dev_f1_tacrev = scorer.score(tacrev_dev_batch.gold(), predictions_tacrev)

        train_loss = (
            train_loss / train_batch.num_examples * opt["batch_size"]
        )  # Avg loss per batch
        dev_loss_tacred = (
            dev_loss_tacred / tacred_dev_batch.num_examples * opt["batch_size"]
        )
        dev_loss_tacrev = (
            dev_loss_tacrev / tacrev_dev_batch.num_examples * opt["batch_size"]
        )
        print(
            "epoch {}: train_loss = {:.6f}, tacred_dev_loss = {:.6f}, tacred_dev_f1 = {:.4f}, tacrev_dev_loss = {:.6f}, tacrev_dev_f1 = {:.4f}".format(
                epoch,
                train_loss,
                dev_loss_tacred,
                dev_f1_tacred,
                dev_loss_tacrev,
                dev_f1_tacrev,
            )
        )
        file_logger.log(
            "{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.6f}\t{:.4f}".format(
                epoch,
                train_loss,
                dev_loss_tacred,
                dev_f1_tacred,
                dev_loss_tacrev,
                dev_f1_tacrev,
            )
        )

        # Save tacred
        model_file = model_save_dir + "/checkpoint_epoch_{}_tacred.pt".format(epoch)
        model.save(model_file, epoch)
        if epoch == 1 or dev_f1_tacred > max(dev_f1_history_tacred):
            copyfile(model_file, model_save_dir + "/best_model_tacred.pt")
            print("new best model saved.")
        if epoch % opt["save_epoch"] != 0:
            os.remove(model_file)

        # Save tacrev
        model_file = model_save_dir + "/checkpoint_epoch_{}_tacrev.pt".format(epoch)
        model.save(model_file, epoch)
        if epoch == 1 or dev_f1_tacrev > max(dev_f1_history_tacrev):
            copyfile(model_file, model_save_dir + "/best_model_tacrev.pt")
            print("new best model saved.")
        if epoch % opt["save_epoch"] != 0:
            os.remove(model_file)

        dev_f1_history_tacred += [dev_f1_tacred]
        dev_f1_history_tacrev += [dev_f1_tacrev]

        print("")

    print("Training ended with {} epochs.".format(epoch))
