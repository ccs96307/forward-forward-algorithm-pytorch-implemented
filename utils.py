# coding: utf-8
import torch


def create_pos_data(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
) -> torch.Tensor:
    return labeled_image(inputs, labels)


def create_neg_data(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
) -> torch.Tensor:
    fake_labels = torch.randint(1, 10, (labels.shape[0],))
    fake_labels = (labels + fake_labels) % num_classes
    return labeled_image(inputs, fake_labels)


def create_test_data(
    inputs: torch.Tensor,
    num_classes: int = 10,
) -> torch.Tensor:
    test_data = torch.stack([labeled_image(inputs, idx) for idx in torch.tensor(range(num_classes))])
    return test_data.view(num_classes, -1)


def labeled_image(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
) -> torch.Tensor:
    # Convert into one-hot format
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)

    # Replace
    # one_hot_labels = (batch_size, 10)
    # images         = (batch_size, 28 * 28)
    images = inputs.clone()
    images[:, :10] = one_hot_labels

    return images


class AverageMeter:
    """Computes and stores the average and current values of losses"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, count_num: int = 1) -> None:
        self.val = val
        self.sum += val * count_num
        self.count += count_num
        self.avg = self.sum / self.count