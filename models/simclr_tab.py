import pytorch_lightning as pl
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad
import torch.nn.functional as F
from models.resnet import ResNet18_Tabular

class ResidualBlock(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.bn1 = nn.BatchNorm1d(feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.bn2 = nn.BatchNorm1d(feature_dim)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual  
        return F.relu(out)
    
class TabularResNet(nn.Module):
    def __init__(self, input_dim, feature_dim, num_blocks=3):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, feature_dim)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(feature_dim) for _ in range(num_blocks)]
        )
    
    def forward(self, x):
        out = F.relu(self.input_fc(x))
        out = self.res_blocks(out)
        return out

class MLP(torch.nn.Sequential):
    def __init__(self, input_dim: int, 
                 hidden_dim: int, 
                 num_hidden: int, 
                 dropout: float = 0.0) -> None:
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class SimCLR_Tab(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['train', 'validate', 'linear_eval']
        self.num_classes = args.num_classes
        self.input_dim = args.input_dim        
        self.feature_dim = args.feature_dim
        self.output_dim = args.output_dim
        self.backbone_n_layer = args.backbone_n_layer
        self.head_n_layer = args.head_n_layer
         
        self.example_input_array = torch.Tensor(1, self.input_dim) # print summary 
              
        self.backbone = MLP(self.input_dim, self.feature_dim, self.backbone_n_layer, dropout=0.2)
        # self.backbone = TabularResNet(self.input_dim, self.feature_dim, self.backbone_n_layer)
        # self.backbone = ResNet18_Tabular()
        self.projection_head = MLP(self.feature_dim, self.output_dim, self.head_n_layer)  
        self.fc = nn.Linear(self.feature_dim, self.num_classes)
        
        self.lr=args.lr
        self.momentum=args.momentum 
        self.weight_decay=args.weight_decay
        self.max_epochs = args.max_epochs
        self.ckpt_path = args.ckpt_path

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
            x0, x1 = batch[0]
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
            x0, x1 = batch[0]
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
        ckpt = torch.load(self.ckpt_path)['state_dict']
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
