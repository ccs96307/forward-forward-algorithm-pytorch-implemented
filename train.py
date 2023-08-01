# coding: utf-8
from sklearn import metrics
import torch
from tqdm import tqdm

from dataloader import get_mnist_dataloader
from ffmodel import FFClassifier
from utils import AverageMeter, create_pos_data, create_neg_data, create_test_data

torch.manual_seed(2999)


def main() -> None:
    # Settings
    num_epochs = 80
    batch_size = 64

    # DataLoader
    train_dataloader = get_mnist_dataloader(_mode="train", batch_size=batch_size)
    val_dataloader = get_mnist_dataloader(_mode="val", batch_size=1)

    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # FFModel
    model = FFClassifier([28*28, 2000, 2000, 2000, 2000], device=device)
    torch.compile(model)
    
    # Loss Logger
    loss_logger = AverageMeter()

    # Train
    for epoch in range(1, num_epochs+1):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Train - Epoch [{epoch}/{num_epochs}] Loss: {loss_logger.avg:.4f}")
        loss_logger.reset()

        for inputs, labels in pbar:
            pos_inputs = create_pos_data(inputs, labels).to(device)
            neg_inputs = create_neg_data(inputs, labels).to(device)

            loss = model(pos_inputs=pos_inputs, neg_inputs=neg_inputs)
            loss_logger.update(loss, inputs.shape[0])
            pbar.set_description(f"Train - Epoch [{epoch}/{num_epochs}] Loss: {loss_logger.avg:.4f}")

        torch.save(model, f"./models/epoch{epoch}.ckpt")

        # Validation
        model.eval()

        # Evaluation
        predicts = []
        targets = []
        for inputs, labels in tqdm(val_dataloader):
            inputs = create_test_data(inputs).to(device)
            predict = model.predict(inputs)
            
            predicts.append(predict.item())
            targets.append(labels.item())

        print(metrics.classification_report(targets, predicts))
        print()


if __name__ == "__main__":
    main()
