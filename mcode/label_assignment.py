import os
from typing import List
import matplotlib.pyplot as plt

import torch


class LabelVis:
    def __init__(self,
                 model,
                 save_path,
                 strategy=None,
                 num_samples=3,
                 img_names: list=None,
                 type='iter',
                 rate=5):
        self.model = model
        self.save_path = os.path.join(save_path, "LabelVis")
        self.strategy = strategy
        self.num_samples = num_samples
        self.img_idxs = [i for i in range(num_samples)]
        self.img_names = img_names
        assert type in ['iter', 'epoch', None]
        self.type = type
        self.rate = rate

    def before_train(self, dataset):
        """ This function gathers images and labels for visualization """
        os.makedirs(self.save_path, exist_ok=True)
        self.images = []
        self.labels = []
        if self.img_names is not None:
            self.img_idxs = []
            for i in range(len(dataset)):
                sample = dataset.images[i]
                img_name = os.path.basename(sample)
                if img_name in self.img_names:
                    self.img_idxs.append(i)
        for idx in self.img_idxs:
            sample = dataset[idx]
            self.images.append(sample['image'])
            self.labels.append(sample['mask'])

    def after_train_iter(self, iter, epoch, strategy_kwargs):
        if self.type == 'iter' and (iter - 1) % self.rate == 0:
            device = next(self.model.parameters()).device
            with torch.no_grad(): # make process not affect model parameters
                for idx, (image, label) in enumerate(zip(
                    self.images, self.labels
                )):
                    image = image[None].to(device)
                    label = label[None].to(device)
                    # --- perform forward pass ---
                    y_hats = self.model(image)
                    # --- perform label assignment ---
                    targets = label_assignment(y_hats,
                                               label,
                                               self.strategy,
                                               **strategy_kwargs)
                    # --- visualize ---
                    fig = plt.figure()
                    plt.axis('off')
                    string = f"Image-{idx}_epoch-{epoch}_iter-{iter}"
                    plt.title(string)
                    rows = len(y_hats)
                    cols = 2
                    for i, (y_hat, target) in enumerate(zip(y_hats,
                                                            targets)):
                        pos = i * 2 + 1
                        res = y_hat.sigmoid()
                        res = (res - res.min())/(res.max() - res.min())
                        res = res[0, 0].cpu().numpy() # shape: (H, W)
                        target = target[0, 0].cpu().numpy()
                        fig.add_subplot(rows, cols, pos)
                        plt.imshow(res, cmap='gray')
                        fig.add_subplot(rows, cols, pos+1)
                        plt.imshow(target, cmap='gray')
                    plt.savefig(f"{self.save_path}/{string}.jpg")
                    plt.show()
                    plt.close()
        else:
            pass

    def after_train_epoch(self, epoch, strategy_kwargs):
        if self.type == 'epoch' and (epoch - 1) % self.rate == 0:
            device = next(self.model.parameters()).device
            with torch.no_grad():  # make process not affect model parameters
                for idx, (image, label) in enumerate(zip(
                        self.images, self.labels
                )):
                    image = image[None].to(device)
                    label = label[None].to(device)
                    # --- perform forward pass ---
                    y_hats = self.model(image)
                    # --- perform label assignment ---
                    targets = label_assignment(y_hats,
                                               label,
                                               self.strategy,
                                               **strategy_kwargs)
                    # --- visualize ---
                    fig = plt.figure()
                    plt.axis('off')
                    string = f"Image-{idx}_epoch-{epoch}"
                    plt.title(string)
                    rows = len(y_hats)
                    cols = 2
                    for i, (y_hat, target) in enumerate(zip(y_hats,
                                                            targets)):
                        pos = i * 2 + 1
                        res = y_hat.sigmoid()
                        res = (res - res.min()) / (res.max() - res.min())
                        res = res[0, 0].cpu().numpy()  # shape: (H, W)
                        target = target[0, 0].cpu().numpy()
                        fig.add_subplot(rows, cols, pos)
                        plt.imshow(res, cmap='gray')
                        fig.add_subplot(rows, cols, pos + 1)
                        plt.imshow(target, cmap='gray')
                    plt.savefig(f"{self.save_path}/{string}.jpg")
                    plt.show()
                    plt.close()
        else:
            pass


def label_assignment(preds: List[torch.Tensor], target: torch.Tensor=None,
                     assign_func=None, **kwargs):
    if assign_func is None:
        return [target] * len(preds)
    else:
        targets = assign_func(preds, target, **kwargs)
        return targets