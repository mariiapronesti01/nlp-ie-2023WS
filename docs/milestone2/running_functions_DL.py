# from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import torch
import time

from utilities_DL import epoch_time


def train(model, iterator, optimizer, criterion):
    # We will calculate loss and accuracy epoch-wise based on average batch accuracy
    epoch_loss = 0

    # You always need to set your model to training mode
    # If you don't set your model to training mode the error won't propagate back to the weights
    model.train()

    # We calculate the error on batches so the iterator will return matrices with shape [BATCH_SIZE, VOCAB_SIZE]
    for batch in iterator:
        text_vecs = batch[0]
        labels = batch[1]
        sen_lens = []
        texts = []

        # This is for later!
        if len(batch) > 2:
            sen_lens = batch[2]
            texts = batch[3]

        # We reset the gradients from the last step, so the loss will be calculated correctly (and not added together)
        optimizer.zero_grad()

        # This runs the forward function on your model (you don't need to call it directly)
        predictions = model(text_vecs, sen_lens)

        # Calculate the loss and the accuracy on the predictions (the predictions are log probabilities, remember!)
        loss = criterion(predictions, labels)

        # prec, recall, fscore = calculate_performance(predictions, labels)

        # Propagate the error back on the model (this means changing the initial weights in your model)
        # Calculate gradients on parameters that requries grad
        loss.backward()
        # Update the parameters
        optimizer.step()

        # We add batch-wise loss to the epoch-wise loss
        epoch_loss += loss.item()
        # We also do the same with the scores
        # epoch_prec += prec.item()
        # epoch_recall += recall.item()
        # epoch_fscore += fscore.item()
        mean_epoch_loss = epoch_loss / len(iterator)
    return mean_epoch_loss
    # epoch_prec / len(iterator),
    # epoch_recall / len(iterator),
    # epoch_fscore / len(iterator),


# The evaluation is done on the validation dataset
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_prec = 0
    epoch_recall = 0
    epoch_fscore = 0
    # On the validation dataset we don't want training so we need to set the model on evaluation mode
    model.eval()

    # Also tell Pytorch to not propagate any error backwards in the model or calculate gradients
    # This is needed when you only want to make predictions and use your model in inference mode!
    with torch.no_grad():
        # The remaining part is the same with the difference of not using the optimizer to backpropagation
        for batch in iterator:
            text_vecs = batch[0]
            labels = batch[1]
            sen_lens = []
            texts = []

            if len(batch) > 2:
                sen_lens = batch[2]
                texts = batch[3]

            predictions = model(text_vecs, sen_lens)
            loss = criterion(predictions, labels)

            # prec, recall, fscore = calculate_performance(predictions, labels)

            epoch_loss += loss.item()
            # epoch_prec += prec.item()
            # epoch_recall += recall.item()
            # epoch_fscore += fscore.item()

    # Return averaged loss on the whole epoch!
    return epoch_loss / len(iterator)
    # epoch_prec / len(iterator),
    # epoch_recall / len(iterator),
    # epoch_fscore / len(iterator),


def training_loop_auto(
    model,
    train_iterator,
    optimizer,
    criterion,
    valid_iterator,
    modelpath,
    epoch_number=15,
    patience=3,
):
    """
    Implements the training loop with automatic early-stopping
    """
    # Set an EPOCH number!
    N_EPOCHS = epoch_number
    consecutive_no_improvement = 0

    best_valid_loss = float("inf")

    # We loop forward on the epoch number
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # Train the model on the training set using the dataloader
        train_loss = train(model, train_iterator, optimizer, criterion)
        # And validate your model on the validation set
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # If we find a better model, we save the weights so later we may want to reload it
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            consecutive_no_improvement = 0
            torch.save(model.state_dict(), modelpath)

        else:
            consecutive_no_improvement += 1

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f}")

        # Check for early stopping
        if consecutive_no_improvement >= patience:
            print(
                f"No improvement in validation loss for {patience} consecutive epochs. Early stopping after epoch {epoch+1}."
            )
            break  # Terminate training loop

    return best_valid_loss


def training_loop(
    model,
    train_iterator,
    optimizer,
    criterion,
    valid_iterator,
    modelpath,
    epoch_number=15,
):
    # Set an EPOCH number!
    N_EPOCHS = epoch_number

    best_valid_loss = float("inf")

    # We loop forward on the epoch number
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # Train the model on the training set using the dataloader
        train_loss = train(model, train_iterator, optimizer, criterion)
        # And validate your model on the validation set
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # If we find a better model, we save the weights so later we may want to reload it
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), modelpath)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f}")
