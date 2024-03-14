import pytorch_lightning as pl
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad
import torch.nn as nn
import torch
from tqdm import tqdm

class Classifier(pl.LightningModule):
    def __init__(self, backbone, args):
        super().__init__()
        self.example_input_array = torch.Tensor(1, 3, 224, 224) # print summary 

        self.backbone = backbone      
        self.fc = nn.Linear(self.backbone.hidden_dim, args.num_classes)
        self.lr=args.lr
        self.mode = args.mode
        assert self.mode in ['train', 'validate']
        self.momentum=args.momentum 
        self.weight_decay=args.weight_decay
        self.max_epochs = args.max_epochs
        self.ckpt_path = args.ckpt_path

        self.criterion = nn.CrossEntropyLoss()
        self.outputs = []
        
        if self.mode == 'validate':
            print(f'validation mode. loading model from {self.ckpt_path}...')
            self._load_weights()
            deactivate_requires_grad(self)

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_cls", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._infer_batch(batch)

    def on_validation_epoch_end(self):
        acc = self._compute_acc()
        self.log("val_acc_cls", acc, on_epoch=True, prog_bar=True)

    # def on_test_start(self):
    #     self._load_weights()

    # def test_step(self, batch, batch_idx):
    #     self._infer_batch(batch)

    # def on_test_epoch_end(self):
    #     acc = self._compute_acc()
    #     self.log("test_acc_cls", acc, on_epoch=True, prog_bar=True)
            
    def _infer_batch(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        self.outputs.append((num, correct))

    def _compute_acc(self):
        total_num = 0
        total_correct = 0
        for num, correct in self.outputs:
            total_num += num
            total_correct += correct
        acc = total_correct / total_num
        self.outputs.clear()
        return acc
    
    def _load_weights(self):
        ckpt = torch.load(self.ckpt_path)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        keys = [key for key in ckpt.keys()  if key.startswith(('layer', 'conv', 'bn'))]       
        for key in keys:
            ckpt[f'backbone.{key}'] = ckpt.pop(key)
        self.load_state_dict(ckpt, strict=False)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=self.lr, 
            momentum=self.momentum, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
    