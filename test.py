# coding: utf-8
from sklearn import metrics
import torch
from tqdm import tqdm

from dataloader import get_mnist_dataloader
from utils import create_test_data

torch.manual_seed(2999)


def main() -> None:
    # DataLoader
    # train_dataloader = get_mnist_dataloader(_mode="train", batch_size=16)
    # val_dataloader = get_mnist_dataloader(_mode="val", batch_size=16)
    test_dataloader = get_mnist_dataloader(_mode="test", batch_size=1)
    
    # FFModel
    # model = FFClassifier([28*28, 2000, 2000, 2000, 2000])
    model = torch.load("./models/epoch80.ckpt")

    # Evaluation
    predicts = []
    targets = []
    for inputs, labels in tqdm(test_dataloader):
        inputs_all_labels = create_test_data(inputs)

        predict = model.predict(inputs_all_labels)
        predicts.append(predict.item())
        targets.append(labels.item())

    print(metrics.classification_report(targets, predicts))


if __name__ == "__main__":
    main()
