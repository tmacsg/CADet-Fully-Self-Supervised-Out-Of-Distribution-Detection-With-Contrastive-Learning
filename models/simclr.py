import pytorch_lightning as pl
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad

class SimCLR(pl.LightningModule):
    def __init__(self, backbone, args):
        super().__init__()
        self.example_input_array = torch.Tensor(1, 3, 600, 800) # print summary 
        self.backbone = backbone

        self.mode = args.mode
        assert self.mode in ['train', 'validate', 'linear_eval']
        self.lr=args.lr
        self.momentum=args.momentum 
        self.weight_decay=args.weight_decay
        self.max_epochs = args.max_epochs
        self.ckpt_path = args.ckpt_path

        self.fc = nn.Linear(self.backbone.hidden_dim, args.num_classes)
        self.projection_head = SimCLRProjectionHead(self.backbone.hidden_dim, 
                                                    self.backbone.hidden_dim, 128,
                                                    num_layers=args.num_layers)
        self.criterion_simclr = NTXentLoss(temperature=0.1)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.outputs = []
        
        if self.mode == 'linear_eval':
            print(f'linear evaluation mode. loading model from {self.ckpt_path}...')
            self._load_weights()
            deactivate_requires_grad(self.backbone)
            activate_requires_grad(self.fc)
        elif self.mode == 'validate':
            print(f'validation mode. loading model from {self.ckpt_path}...')
            self._load_weights()
            deactivate_requires_grad(self)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        if self.mode == 'train':
            (x0, x1), _, _ = batch
            z0 = self.forward(x0)
            z1 = self.forward(x1)
            loss = self.criterion_simclr(z0, z1)
            self.log("train_loss_ssl", loss, prog_bar=True, batch_size=len(x0))
            return loss
        elif self.mode == 'linear_eval':
            x, y = batch
            y_hat = self.fc(self.backbone(x))
            loss = self.criterion_cls(y_hat, y)
            self.log("train_loss_cls", loss, prog_bar=True)
            return loss
    
    def validation_step(self, batch, batch_idx):
        if self.mode == 'train':
            (x0, x1), _, _ = batch
            z0 = self.forward(x0)
            z1 = self.forward(x1)
            loss = self.criterion_simclr(z0, z1)
            self.log("val_loss_ssl", loss, prog_bar=True, batch_size=len(x0))
            return loss
        elif self.mode == 'linear_eval' or self.mode == 'validate':
            self._infer_batch(batch)
        
    def on_validation_epoch_end(self):
        if self.mode == 'linear_eval' or self.mode == 'validate':
            acc = self._compute_acc()
            self.log("val_acc_cls", acc, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        self._infer_batch(batch)

    def on_test_epoch_end(self):
        acc = self._compute_acc()
        self.log("test_acc_cls", acc, on_epoch=True, prog_bar=True)
    
    def _infer_batch(self, batch):
        x, y = batch
        y_hat = self.fc(self.backbone(x))
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
        print('load weights from: ', self.ckpt_path)
        ckpt = torch.load(self.ckpt_path)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        keys = [key for key in ckpt.keys()  if key.startswith('head')]
        for key in keys:
            ckpt[key[5:]] = ckpt.pop(key)
        self.load_state_dict(ckpt, strict=False)
             
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=self.lr, 
            momentum=self.momentum, 
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
