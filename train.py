# coding: utf-8
import torch
from tqdm import tqdm

from dataloader import get_mnist_dataloader
from ffmodel import FFClassifier
from utils import AverageMeter, create_pos_data, create_neg_data

torch.manual_seed(2999)


def main() -> None:
    # Settings
    num_epochs = 80
    batch_size = 64

    # DataLoader
    train_dataloader = get_mnist_dataloader(_mode="train", batch_size=batch_size)
    val_dataloader = get_mnist_dataloader(_mode="val", batch_size=batch_size)
    test_dataloader = get_mnist_dataloader(_mode="test", batch_size=batch_size)
    
    # FFModel
    model = FFClassifier([28*28, 2000, 2000, 2000, 2000])

    # Loss Logger
    loss_logger = AverageMeter()

    # Train
    for epoch in range(1, num_epochs+1):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Train - Epoch [{epoch}/{num_epochs}] Loss: {loss_logger.avg:.4f}")
        loss_logger.reset()

        for inputs, labels in pbar:
            pos_inputs = create_pos_data(inputs, labels)
            neg_inputs = create_neg_data(inputs, labels)

            loss = model(pos_inputs, neg_inputs)
            loss_logger.update(loss, inputs.shape[0])
            pbar.set_description(f"Train - Epoch [{epoch}/{num_epochs}] Loss: {loss_logger.avg:.4f}")

        torch.save(model, f"./models/epoch{epoch}.ckpt")

        # Validation
        model.eval()
        pbar = tqdm(val_dataloader, desc=f"Valid - Epoch [{epoch}/{num_epochs}] Loss: {loss_logger.avg:.4f}")
        loss_logger.reset()

        for inputs, labels in pbar:
            pos_inputs = create_pos_data(inputs, labels)
            neg_inputs = create_neg_data(inputs, labels)

            with torch.no_grad():
                loss = model(pos_inputs, neg_inputs, train_mode=False)

            loss_logger.update(loss, inputs.shape[0])
            pbar.set_description(f"Valid - Epoch [{epoch}/{num_epochs}] Loss: {loss_logger.avg:.4f}")

        print()

if __name__ == "__main__":
    main()
