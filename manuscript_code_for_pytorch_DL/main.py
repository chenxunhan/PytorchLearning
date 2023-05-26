import os
import torch
import datetime
from torch import nn
from model.Classic_NN import NeuralNetwork
from data.dataset import get_my_dataset
from data.dataloader import get_my_dataloader
from tool.utils import get_specific_time
from tool.logger import *

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    logger.info(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def eval(model, test_data):
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        # print(f'Predicted: "{predicted}", Actual: "{actual}"')
        logger.info(f'Predicted: "{predicted}", Actual: "{actual}"')

def main():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")
    logger.info(f"Using {device} device")

    # Create data loaders.
    training_data, test_data, num_examples = get_my_dataset()
    # print(f"Data Info: {num_examples}")
    logger.info(f"Data Info: {num_examples}")

    batch_size = 512
    do_training = True
    do_shuffle = False

    test_dataloader = get_my_dataloader(test_data, batch_size, do_shuffle)

    # Create the criterion
    loss_fn = nn.CrossEntropyLoss()

    if do_training:
        do_shuffle = True
        train_dataloader = get_my_dataloader(training_data, batch_size, do_shuffle)

        # Create neural network model.
        my_model = NeuralNetwork().to(device)
        # print(f"Model Info:\n {my_model}")
        logger.info(f"Model Info:\n {my_model}")

        # Create the optimization method
        optimizer = torch.optim.SGD(my_model.parameters(), lr=1e-3)

        # Create the training loop
        epochs = 10
        for t in range(epochs):
            # print(f"Epoch {t + 1}\n-------------------------------")
            logger.info(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, my_model, loss_fn, optimizer, device)

        # print("Training Done!")
        logger.info("Training Done!")

        # Save the training model
        # save_path = "./save_path/Classic_NN_" + get_specific_time() + ".pth"
        save_path = "./save_path/Classic_NN.pth"
        torch.save(my_model.state_dict(), save_path)
        # print("Saved PyTorch Model State to save_path")
        logger.info("Saved PyTorch Model State to save_path")

    # loading the training model
    my_model = NeuralNetwork().to(device)
    # my_model.load_state_dict(torch.load("./save_path/Classic_NN_2022_11_1_15h39m5s.pth"))
    my_model.load_state_dict(torch.load("./save_path/Classic_NN.pth"))

    # Create the test process
    test(test_dataloader, my_model, loss_fn, device)
    eval(my_model, test_data)

if __name__ == '__main__':
    # Create the log
    LOG_PATH = "./log_path"
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    try:
        main()
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        # save_model(model, os.path.join(train_dir, "earlystop"))
    except Exception as e:
        logger.error("[Error] Other Error. Stoping")
        logger.error(e)
        # save_model(model, os.path.join(train_dir, "OtherError"))

