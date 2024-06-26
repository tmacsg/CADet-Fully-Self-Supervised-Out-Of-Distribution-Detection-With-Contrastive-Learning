from typing import Tuple, Union, List, Optional, Any
from PIL import Image
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, RandomSampler, BatchSampler, Subset
from torchvision.datasets import CIFAR10, ImageFolder, MNIST
from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms import SimCLRTransform, SimCLRViewTransform
import torchvision.transforms as T
import numpy as np 
import os
import torch
import re
from tqdm import tqdm
import random
import pickle
from sklearn.preprocessing import StandardScaler
from torch.distributions.uniform import Uniform
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from scipy.io import arff

class CADetTransform:
    def __init__(
        self,
        num_tranforms: int
    ):
        self.n_transforms = num_tranforms
        self.view_transform = T.Compose([
            Convert('RGB'),
            T.Resize([256,256], interpolation=T.InterpolationMode.BILINEAR),
            T.RandomResizedCrop(size=224, scale=(0.75, 0.75)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
        ])

    def __call__(self, image):
        return torch.stack([self.view_transform(image) for _ in range(self.n_transforms)])
    
class CADetTransform_CIFAR:
    def __init__(
        self,
        num_tranforms: int,
    ):
        self.view_transform = T.Compose([
            Convert('RGB'),
            T.Resize([32,32], interpolation=T.InterpolationMode.BILINEAR),
            T.RandomResizedCrop(size=32, scale=(0.75, 0.75)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
        ])
        self.n_transforms = num_tranforms
        # self.view_transform = SimCLRViewTransform(input_size=32, min_scale=0.75, cj_prob=0)

    def __call__(self, image):
         return torch.stack([self.view_transform(image) for _ in range(self.n_transforms)])
         
class PixelFlick:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image):
        image = np.array(image, dtype=np.uint8)
        noise = torch.randint(-self.scale, self.scale, image.shape)
        image =  np.clip(image + noise.numpy(), 0, 255).astype(np.uint8)
        return Image.fromarray(image)

class Random_Noise:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, X):
        noise = np.random.normal(0, self.sigma, len(X)).astype(np.float32)
        return X + noise
    
class Random_Noise_Mul:
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, X):
        noise = np.random.normal(1, self.sigma, len(X)).astype(np.float32)
        return X * noise
    
class Random_Mask:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, X):
        mask = np.random.binomial(1, 1-self.p, size=len(X)).astype(np.float32)
        return X * mask
class L2_Normalize: 
    def __call__(self, X):
        norm = torch.norm(X) + 1e-8
        return X / norm
    
class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)
    
class ToTensor:
    def __call__(self, X):
        return torch.tensor(X)

class CIFAR10_NPY(Dataset):
    """ load cifar dataset from .npy format
    """
    def __init__(self, root, transform=None):
        self.transform = transform
        _data = np.load(root, allow_pickle=True)
        if _data.dtype != 'uint8':
            # dict
            self.data = _data.item()['data']
            self.targets = _data.item()['labels']
        else:
            self.data = _data
            self.targets = np.full((len(self.data),), 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.targets[index]
        sample = T.ToPILImage()(sample)     
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label
    
class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.cifar10_path = args.cifar10_path
        self.cifar10_1_path = args.cifar10_1_path
        self.cifar10_fgsm_path = args.cifar10_fgsm_path
        self.cifar10_cw_path = args.cifar10_cw_path
        self.cifar10_pgd_path = args.cifar10_pgd_path
        self.mode = args.mode
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        assert self.mode in ['supervised', 'unsupervised', 'mmd', 'cadet', 'mmd_ss']
        self.mmd_sample_sizes = args.mmd_sample_sizes
        self.mmd_n_tests = args.mmd_n_tests
        self.mmd_image_set_q = args.mmd_image_set_q
        assert self.mmd_image_set_q in ['same_dist', 'cifar10_1', 'pgd', 'cw', 'fgsm']
        self.cadet_n_tests = args.cadet_n_tests
        self.cadet_n_transforms = args.cadet_n_transforms
        self.test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])])

    def prepare_data(self):
        CIFAR10(root=self.cifar10_path, train=True, download=True)
        CIFAR10(root=self.cifar10_path, train=False, download=True)
        
    def setup(self, stage: str):
        if self.mode == 'supervised':
            if stage == 'fit':
                # self.train_dataset = CIFAR10(root=self.cifar10_path, train=True, download=True,
                #                              transform=T.Compose([T.RandomCrop(32, padding=4), 
                #                                                  T.RandomHorizontalFlip(),
                #                                                  T.ToTensor()])) 
                self.train_dataset = CIFAR10(root=self.cifar10_path, train=True, download=True,
                                             transform=SimCLRViewTransform(input_size=32)) 
                self.val_dataset = CIFAR10(root=self.cifar10_path, train=False, download=True, transform=self.test_transform)
            if stage == 'test' or stage == 'validate': 
                # self.val_dataset = CIFAR10(root=self.cifar10_path, train=False, download=True, transform=self.test_transform)
                self.val_dataset = CIFAR10(root=self.cifar10_path, train=False, download=True, transform=SimCLRViewTransform(input_size=32))
                # self.val_dataset = CIFAR10(root=self.cifar10_path, train=False, download=True, transform=self.test_transform)
                # self.val_dataset = CIFAR10_NPY(root=self.cifar10_fgsm_path, transform=self.test_transform)
                # self.val_dataset = CIFAR10_NPY(root=self.cifar10_cw_path, transform=self.test_transform)
                # self.val_dataset = CIFAR10_NPY(root=self.cifar10_pgd_path, transform=self.test_transform)
                # self.val_dataset = CIFAR10(root=self.cifar10_path, train=True, download=True, transform=T.ToTensor())

        if self.mode == 'unsupervised':
            if stage == 'fit':
                train_dataset = CIFAR10(root=self.cifar10_path, train=True, download=True,
                                  transform=SimCLRTransform(input_size=32))
                val_dataset = CIFAR10(root=self.cifar10_path, train=False, download=True,
                                  transform=SimCLRTransform(input_size=32))
                self.train_dataset = LightlyDataset.from_torch_dataset(train_dataset)
                self.val_dataset = LightlyDataset.from_torch_dataset(val_dataset)

        if self.mode == 'mmd':
            if stage == 'test':
                self.dataset_s = CIFAR10(root=self.cifar10_path, train=False, download=True, transform=self.test_transform)
                if self.mmd_image_set_q == 'same_dist':
                    self.dataset_q = CIFAR10(root=self.cifar10_path, train=True, download=True, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'cifar10_1':
                    self.dataset_q = CIFAR10_NPY(root=self.cifar10_1_path, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'pgd':
                    self.dataset_q = CIFAR10_NPY(root=self.cifar10_pgd_path, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'cw':
                    self.dataset_q = CIFAR10_NPY(root=self.cifar10_cw_path, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'fgsm':
                    self.dataset_q = CIFAR10_NPY(root=self.cifar10_fgsm_path, transform=self.test_transform) 

        if self.mode == 'mmd_ss':
            if stage == 'test':
                test_transform_s = CADetTransform_CIFAR(num_tranforms=max(self.mmd_sample_sizes))
                test_transform_q = CADetTransform_CIFAR(num_tranforms=max(self.mmd_sample_sizes))                         
                self.dataset_s = CIFAR10(root=self.cifar10_path, train=False, download=True, transform=test_transform_s)
                if self.mmd_image_set_q == 'same_dist':
                    self.dataset_q = CIFAR10(root=self.cifar10_path, train=False, download=True, transform=test_transform_q) 
                elif self.mmd_image_set_q == 'cifar10_1':
                    self.dataset_q = CIFAR10_NPY(root=self.cifar10_1_path, transform=test_transform_q) 
                elif self.mmd_image_set_q == 'pgd':
                    self.dataset_q = CIFAR10_NPY(root=self.cifar10_pgd_path, transform=test_transform_q) 
                elif self.mmd_image_set_q == 'cw':
                    self.dataset_q = CIFAR10_NPY(root=self.cifar10_cw_path, transform=test_transform_q) 
                elif self.mmd_image_set_q == 'fgsm':
                    self.dataset_q = CIFAR10_NPY(root=self.cifar10_fgsm_path, transform=test_transform_q) 

        if self.mode == 'cadet':
            if stage == 'test':
                transform = CADetTransform_CIFAR(num_tranforms=self.cadet_n_transforms)
                self.test_dataset_cifar10_1 = CIFAR10_NPY(root=self.cifar10_1_path, transform=transform)
                # self.test_dataset_cifar10_1 = ImageFolder('//wsl.localhost/Ubuntu/home/bingbing/codes/OpenOOD/data/images_classic/cifar100/test',  transform=transform)
                self.test_dataset_pgd = CIFAR10_NPY(root=self.cifar10_pgd_path, transform=transform)
                self.test_dataset_cw =CIFAR10_NPY(root=self.cifar10_cw_path, transform=transform) 
                self.test_dataset_fgsm = CIFAR10_NPY(root=self.cifar10_fgsm_path, transform=transform) 
                
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, 
                              pin_memory=True, drop_last=True, shuffle=True, persistent_workers=True) 
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, 
                            pin_memory=True, drop_last=True, shuffle=False, persistent_workers=True) 
        # return DataLoader(self.val_dataset, batch_size=3, num_workers=8, collate_fn=self.mmd_ss_collate_fn,
        #                     pin_memory=True, drop_last=True, shuffle=False, persistent_workers=True) 
        
    def test_dataloader(self):
        if self.mode == 'supervised':
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, 
                            pin_memory=True, drop_last=True, shuffle=False, persistent_workers=True) 
        
        if self.mode == 'mmd':
            batch_size = max(self.mmd_sample_sizes)
            num_samples = batch_size * self.mmd_n_tests
            sampler_s = RandomSampler(self.dataset_s, replacement=False, num_samples=3*num_samples)
            sampler_q = RandomSampler(self.dataset_q, replacement=False, num_samples=num_samples)
            dataloader_s = DataLoader(self.dataset_s, batch_size=batch_size*3, sampler=sampler_s)
            dataloader_q = DataLoader(self.dataset_q, batch_size=batch_size, sampler=sampler_q)
            
            return CombinedLoader({'s': dataloader_s, 'q': dataloader_q})
        
        if self.mode == 'mmd_ss':
            # mmd test with single sample                          
            indexs = [[] for _ in range(self.num_classes)]  
                  
            for idx, (_, class_idx) in enumerate(self.dataset_s):
                indexs[class_idx].append(idx)
            
            sample_indexes_s = []
            for _ in range(self.mmd_n_tests):
                for i in range(self.num_classes):
                    sample_indexes_s.extend(random.choices(indexs[i], k=3))    
          
            indexs_q = random.choices(range(len(self.dataset_q)), k=self.mmd_n_tests)
            sample_indexes_q = list(np.repeat(indexs_q, self.num_classes))
            
            sampler_s = BatchSampler(sample_indexes_s, batch_size=3, drop_last=False)    
            sampler_q = BatchSampler(sample_indexes_q, batch_size=1, drop_last=False)   
            
            dataloader_s = DataLoader(self.dataset_s, batch_sampler=sampler_s, collate_fn=self.mmd_ss_collate_fn)
            dataloader_q = DataLoader(self.dataset_q, batch_sampler=sampler_q, collate_fn=self.mmd_ss_collate_fn)
            
            return CombinedLoader({'s': dataloader_s, 'q': dataloader_q})
        
        if self.mode == 'cadet':
            sampler_test_cifar10_1 = RandomSampler(self.test_dataset_cifar10_1, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_pgd = RandomSampler(self.test_dataset_pgd, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_cw = RandomSampler(self.test_dataset_cw, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_fgsm = RandomSampler(self.test_dataset_fgsm, replacement=False, num_samples=self.cadet_n_tests)
            
            dataloader_test_cifar10_1 = DataLoader(self.test_dataset_cifar10_1, batch_size=1, sampler=sampler_test_cifar10_1)
            dataloader_test_pgd = DataLoader(self.test_dataset_pgd, batch_size=1, sampler=sampler_test_pgd)
            dataloader_test_cw = DataLoader(self.test_dataset_cw, batch_size=1, sampler=sampler_test_cw)
            dataloader_test_fgsm = DataLoader(self.test_dataset_fgsm, batch_size=1, sampler=sampler_test_fgsm)
            
            return CombinedLoader({'cifar10_1': dataloader_test_cifar10_1, 'pgd': dataloader_test_pgd, 
                                   'cw': dataloader_test_cw, 'fgsm': dataloader_test_fgsm})

    
    @staticmethod
    def mmd_ss_collate_fn(batch):
        X, y = [], [] 
        for i in range(len(batch)):
            temp = torch.stack(batch[i][0])
            X.append(temp)
            y.append(torch.tensor(batch[i][1]).expand(len(temp)))
        return torch.cat(X), torch.cat(y)

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_path = args.train_path
        self.val_path = args.val_path
        self.imagenet_o_path = args.imagenet_o_path
        self.inaturalist_path = args.inaturalist_path
        self.pgd_path = args.pgd_path
        self.cw_path = args.cw_path
        self.fgsm_path = args.fgsm_path

        self.mode = args.mode
        assert self.mode in ['supervised', 'unsupervised', 'mmd', 'cadet', 'mmd_ss']
        self.bacth_size = args.batch_size
        self.mmd_sample_sizes = args.mmd_sample_sizes
        self.mmd_n_tests = args.mmd_n_tests
        self.mmd_image_set_q = args.mmd_image_set_q
        assert self.mmd_image_set_q in ['same_dist', 'inaturalist', 'imagenet_o', 'pgd', 'cw', 'fgsm']

        self.cadet_n_tests = args.cadet_n_tests
        self.cadet_n_transforms = args.cadet_n_transforms

        self.normalize = T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"])
        self.test_transform = T.Compose([T.Resize([256,256]),
                                        T.CenterCrop(size=224), 
                                        T.ToTensor(),
                                        self.normalize])
        
    def setup(self, stage: str):
        if self.mode == 'supervised':
            if stage == 'fit':
                self.train_dataset = ImageFolder(root=self.train_path, transform=T.Compose([T.RandomResizedCrop(224),
                                                                                            T.RandomHorizontalFlip(),
                                                                                            T.ToTensor(),
                                                                                             self.normalize]))
                self.val_dataset = ImageFolder(root=self.val_path, transform=self.test_transform)

            if stage == 'test' or stage == 'validate':
                # transform = SimCLRViewTransform()
                self.val_dataset = ImageFolder(root=self.val_path, transform=self.test_transform)   
                # self.val_dataset = ImageFolder(root=self.fgsm_path, transform=transform)  
                # self.val_dataset = ImageFolder(root=self.pgd_path, transform=transform)
                # self.val_dataset = ImageFolder(root=self.cw_path, transform=self.test_transform) 
                # self.val_dataset = ImageFolder(root=self.imagenet_o_path, transform=self.test_transform) 

        if self.mode == 'unsupervised':
            if stage == 'fit':
                train_dataset = ImageFolder(root=self.train_path, transform=SimCLRTransform())                                
                val_dataset = ImageFolder(root=self.val_path, transform=SimCLRTransform())
                self.train_dataset = LightlyDataset.from_torch_dataset(train_dataset)
                self.val_dataset = LightlyDataset.from_torch_dataset(val_dataset)

        if self.mode == 'mmd':
            if stage == 'test':
                self.test_dataset_s = ImageFolder(root=self.val_path, transform=self.test_transform) 
                if self.mmd_image_set_q == 'same_dist':
                    self.test_dataset_q = ImageFolder(root=self.train_path, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'imagenet_o':
                    self.test_dataset_q = ImageFolder(root=self.imagenet_o_path, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'inaturalist':
                    self.test_dataset_q = ImageFolder(root=self.inaturalist_path, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'pgd':
                    self.test_dataset_q = ImageFolder(root=self.pgd_path, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'cw':
                    self.test_dataset_q = ImageFolder(root=self.cw_path, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'fgsm':
                    self.test_dataset_q = ImageFolder(root=self.fgsm_path, transform=self.test_transform) 
                    
        if self.mode == 'mmd_ss':
            if stage == 'test':
                test_transform_q = CADetTransform_CIFAR(num_tranforms=max(self.mmd_sample_sizes))          
                self.dataset_s = ImageFolder(root=self.val_path, transform=self.test_transform) 
                if self.mmd_image_set_q == 'same_dist':
                    self.dataset_q = ImageFolder(root=self.train_path, transform=test_transform_q) 
                elif self.mmd_image_set_q == 'imagenet_o':
                    self.dataset_q = ImageFolder(root=self.imagenet_o_path, transform=test_transform_q) 
                elif self.mmd_image_set_q == 'inaturalist':
                    self.dataset_q = ImageFolder(root=self.inaturalist_path, transform=test_transform_q) 
                elif self.mmd_image_set_q == 'pgd':
                    self.dataset_q = ImageFolder(root=self.pgd_path, transform=test_transform_q) 
                elif self.mmd_image_set_q == 'cw':
                    self.dataset_q = ImageFolder(root=self.cw_path, transform=test_transform_q) 
                elif self.mmd_image_set_q == 'fgsm':
                    self.dataset_q = ImageFolder(root=self.fgsm_path, transform=test_transform_q) 
               
        if self.mode == 'cadet':
            if stage == 'test':
                self.test_dataset_imagenet_o = ImageFolder(root=self.imagenet_o_path, transform=CADetTransform(num_tranforms=self.cadet_n_transforms))
                self.test_dataset_inaturalist = ImageFolder(root=self.inaturalist_path, transform=CADetTransform(num_tranforms=self.cadet_n_transforms))
                self.test_dataset_pgd = ImageFolder(root=self.pgd_path, transform=CADetTransform(num_tranforms=self.cadet_n_transforms))
                self.test_dataset_cw = ImageFolder(root=self.cw_path, transform=CADetTransform(num_tranforms=self.cadet_n_transforms))
                self.test_dataset_fgsm = ImageFolder(root=self.fgsm_path, transform=CADetTransform(num_tranforms=self.cadet_n_transforms))
 
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.bacth_size, num_workers=8,
                          pin_memory=True, drop_last=True, shuffle=True, persistent_workers=True) 
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.bacth_size, num_workers=8, 
                          pin_memory=True, drop_last=True, shuffle=False, persistent_workers=True) 
        
    def test_dataloader(self):
        if self.mode == 'supervised':
            return DataLoader(self.val_dataset, batch_size=self.bacth_size, num_workers=8, 
                              pin_memory=True, drop_last=True, shuffle=False, persistent_workers=True) 
        if self.mode == 'mmd':
            batch_size = max(self.mmd_sample_sizes)
            num_samples = batch_size * self.mmd_n_tests

            sampler_test_s = RandomSampler(self.test_dataset_s, replacement=True, num_samples=3*num_samples)
            sampler_test_q = RandomSampler(self.test_dataset_q, replacement=True, num_samples=3*num_samples)

            dataloader_test_s= DataLoader(self.test_dataset_s, batch_size=batch_size*3, sampler=sampler_test_s,
                                    num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
            dataloader_test_q = DataLoader(self.test_dataset_q, batch_size=batch_size*3, sampler=sampler_test_q,
                                    num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True) 
            return CombinedLoader({'s': dataloader_test_s, 'q': dataloader_test_q})
        
        if self.mode == 'mmd_ss':
            # mmd test with single sample 
            batch_size = max(self.mmd_sample_sizes)
            num_samples = batch_size * self.mmd_n_tests
            sampler_s = RandomSampler(self.dataset_s, replacement=True, num_samples=3*num_samples)
            sampler_q = RandomSampler(self.dataset_q, replacement=True, num_samples=self.mmd_n_tests)
            dataloader_s = DataLoader(self.dataset_s, batch_size=batch_size*3, sampler=sampler_s)
            dataloader_q = DataLoader(self.dataset_q, batch_size=1, sampler=sampler_q, collate_fn=self.mmd_ss_collate_fn)
            
            return CombinedLoader({'s': dataloader_s, 'q': dataloader_q})
        
        if self.mode == 'cadet':
            sampler_test_imagenet_o = RandomSampler(self.test_dataset_imagenet_o, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_inaturalist = RandomSampler(self.test_dataset_inaturalist, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_pgd = RandomSampler(self.test_dataset_pgd, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_cw = RandomSampler(self.test_dataset_cw, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_fgsm = RandomSampler(self.test_dataset_fgsm, replacement=False, num_samples=self.cadet_n_tests)

            dataloader_test_imagenet_o = DataLoader(self.test_dataset_imagenet_o, batch_size=1, sampler=sampler_test_imagenet_o)
            dataloader_test_inaturalist = DataLoader(self.test_dataset_inaturalist, batch_size=1, sampler=sampler_test_inaturalist)
            dataloader_test_pgd = DataLoader(self.test_dataset_pgd, batch_size=1, sampler=sampler_test_pgd)
            dataloader_test_cw = DataLoader(self.test_dataset_cw, batch_size=1, sampler=sampler_test_cw)
            dataloader_test_fgsm = DataLoader(self.test_dataset_fgsm, batch_size=1, sampler=sampler_test_fgsm)
            
            return CombinedLoader({'imagenet_o': dataloader_test_imagenet_o, 'inaturalist': dataloader_test_inaturalist, 
                                   'pgd': dataloader_test_pgd, 'cw': dataloader_test_cw, 'fgsm': dataloader_test_fgsm}, 'max_size_cycle')

   
    @staticmethod
    def mmd_ss_collate_fn(batch):
        X, labels = batch[0]
        if isinstance(X, list):
            X = torch.stack(X)
        return X, labels
    
class AttackDataset(ImageFolder):
    def __init__(self, image_path, transform):
        super().__init__(image_path, transform=transform)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        image_size = sample.size
        image_name = re.split("/|\\\\", path)[-1]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, torch.tensor(image_size), image_name

class DatasetAttacker:
    def __init__(self, image_path, target_path, attacker, normalize=False, device=torch.device('cpu'), input_size=[224,224]):
        self.image_path = image_path
        self.target_path = target_path
        self.attacker = attacker
        self.device = device
        self.normalize = normalize
        self.mean = IMAGENET_NORMALIZE['mean']
        self.std = IMAGENET_NORMALIZE['std']
        transforms = [T.Resize(size=input_size, antialias=True), T.ToTensor()]
        if self.normalize:
            transforms.append(T.Normalize(mean=self.mean, std=self.std))
            self.attacker.set_normalization_used(mean=self.mean, std=self.std)
        
        self.pre_attack_transform = T.Compose(transforms)
        self.dataset = AttackDataset(self.image_path, transform=self.pre_attack_transform)
                
    def attack(self, num_samples = None, batch_size=128):  
        sampler = None if num_samples is None else RandomSampler(self.dataset, num_samples=num_samples) 
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
        for _, (images, labels, image_sizes, image_names) in enumerate(tqdm(self.dataloader)):
            adv_images, labels = images.to(self.device), labels.to(self.device)
            adv_images = self.attacker(images, labels)
            self.save(adv_images, labels, image_sizes, image_names)
                               
    def save(self, images, labels, image_sizes, image_names):
        for image, label, image_size, image_name in zip(images, labels, image_sizes, image_names):
            if self.normalize:
                image = self.attacker.inverse_normalize(image)
            x = self._to_pil_image(image)
            # import matplotlib.pyplot as plt
            # plt.imshow(x)
            # plt.show()
            x = x.resize(image_size)
            class_name = self.dataset.idx_to_class[label.cpu().item()]
            target_path = os.path.join(self.target_path, class_name)
            os.makedirs(target_path, exist_ok=True)
            x.save(os.path.join(target_path, image_name), quality=100)
            
    def _to_pil_image(self, x):
        return T.ToPILImage()(x)
class DatasetAttacker_NoResize:
    def __init__(self, image_path, target_path, attacker, normalize=False, device=torch.device('cpu')):
        self.image_path = image_path
        self.target_path = target_path
        self.attacker = attacker
        self.device = device
        self.normalize = normalize
        self.mean = IMAGENET_NORMALIZE['mean']
        self.std = IMAGENET_NORMALIZE['std']
        transforms = [T.ToTensor()]
        if self.normalize:
            transforms.append(T.Normalize(mean=self.mean, std=self.std))
            self.attacker.set_normalization_used(mean=self.mean, std=self.std)
        
        self.pre_attack_transform = T.Compose(transforms)
        self.dataset = AttackDataset(self.image_path, transform=self.pre_attack_transform)
                
    def attack(self, num_samples = None):  
        sampler = None if num_samples is None else RandomSampler(self.dataset, num_samples=num_samples) 
        self.dataloader = DataLoader(self.dataset, batch_size=1, sampler=sampler, shuffle=False)
        for _, (images, labels, image_sizes, image_names) in enumerate(tqdm(self.dataloader)):
            images, labels = images.to(self.device), labels.to(self.device)
            adv_images = self.attacker(images, labels)
            self.save(adv_images, labels, image_sizes, image_names)
                               
    def save(self, images, labels, image_sizes, image_names):
        for image, label, _, image_name in zip(images, labels, image_sizes, image_names):
            if self.normalize:
                image = self.attacker.inverse_normalize(image).squeeze(0)
            x = self._to_pil_image(image)
            # import matplotlib.pyplot as plt
            # plt.imshow(x)
            # plt.show()
            class_name = self.dataset.idx_to_class[label.cpu().item()]
            target_path = os.path.join(self.target_path, class_name)
            os.makedirs(target_path, exist_ok=True)
            x.save(os.path.join(target_path, image_name), quality=100)
        
    def _to_pil_image(self, x):
        return T.ToPILImage()(x)

class DatasetAttacker_CIFAR10:
    def __init__(self, image_path, target_path, attacker, normalize=False, device=torch.device('cpu')):
        self.image_path = image_path
        self.target_path = target_path
        self.attacker = attacker
        self.normalize = normalize
        self.device = device
        self.mean = IMAGENET_NORMALIZE['mean']
        self.std = IMAGENET_NORMALIZE['std']
        transforms = [T.ToTensor()]
        if self.normalize:
            transforms.append(T.Normalize(mean=self.mean, std=self.std))
            self.attacker.set_normalization_used(mean=self.mean, std=self.std)
        self.dataset = CIFAR10(self.image_path, train=False, 
                               transform=T.Compose(transforms))
        
    def attack(self):   
        all_images = []
        all_labels = []
        self.dataloader = DataLoader(self.dataset, batch_size=256, shuffle=False)
        for _, (images, labels) in enumerate(tqdm(self.dataloader)):
            adv_images, labels = images.to(self.device), labels.to(self.device)
            adv_images = self.attacker(images, labels)
            all_images.append(adv_images)
            all_labels.append(labels)

        all_images, all_labels = torch.cat(all_images,axis=0), torch.cat(all_labels,axis=0)
        if self.normalize:
            all_images = self.attacker.inverse_normalize(all_images)
        all_images = [np.array(self._to_pil_image(x)) for x in all_images]  # torch.vmap error, cannot vectorize!

        data_dict = {
            'data': np.stack(all_images),
            'labels': all_labels.cpu().numpy()
        }
        np.save(self.target_path, data_dict)
                            
    def _to_pil_image(self, x):
        return T.ToPILImage()(x)
     
class Cifar10_ByClass:
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_idxes = list(range(10))
        
    def get_dataset_by_class_id(self, class_id):
        # idx = np.array(self.dataset.targets) == class_id
        idx = np.where(np.array(self.dataset.targets) == class_id)[0]
        return Subset(self.dataset, idx)
    
class MNIST_FAKE(Dataset):
    """ load mnist dataset from .pkl format
    """
    def __init__(self, root, transform=None):
        self.transform = transform
        _data = pickle.load(open(root, 'rb'))[0]        
        self.data = _data
        self.targets = np.full((len(self.data),), 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.targets[index]
        sample = T.ToPILImage()(torch.tensor(sample)) 
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label  

class ContrastiveTransform:
    def __init__(self, view_transform, n_trans = 2):
        self.view_tranform = view_transform
        self.n_trans = n_trans

    def __call__(self, x):
        return [self.view_tranform(x) for _ in range(self.n_trans)]

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mnist_path = args.mnist_path
        self.mnist_path_fake = args.mnist_path_fake
 
        self.mode = args.mode
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        assert self.mode in ['supervised', 'unsupervised', 'mmd']
        self.mmd_sample_sizes = args.mmd_sample_sizes
        self.mmd_n_tests = args.mmd_n_tests
        self.mmd_image_set_q = args.mmd_image_set_q
        assert self.mmd_image_set_q in ['same_dist', 'mnist_fake']

        self.view_transform = T.Compose([Convert('RGB'), SimCLRViewTransform(input_size=32, cj_prob=0)])
        self.contrastive_transform = ContrastiveTransform(self.view_transform)
        self.test_transform = T.Compose([Convert('RGB'), T.ToTensor(), T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])])

    def prepare_data(self):
        MNIST(root=self.mnist_path, train=True, download=True)
        MNIST(root=self.mnist_path, train=False, download=True)
        
    def setup(self, stage: str):
        if self.mode == 'supervised':
            if stage == 'fit':
                self.train_dataset = MNIST(root=self.mnist_path, train=True, download=True, transform=self.view_transform) 
                self.val_dataset = MNIST(root=self.mnist_path, train=False, download=True, transform=self.test_transform)
            if stage == 'test' or stage == 'validate': 
                self.val_dataset = MNIST(root=self.mnist_path, train=False, download=True, transform=self.view_transform)
        if self.mode == 'unsupervised':
            if stage == 'fit':
                train_dataset = MNIST(root=self.mnist_path, train=True, download=True, transform=self.contrastive_transform)
                val_dataset = MNIST(root=self.mnist_path, train=False, download=True, transform=self.contrastive_transform)
                self.train_dataset = LightlyDataset.from_torch_dataset(train_dataset)
                self.val_dataset = LightlyDataset.from_torch_dataset(val_dataset)

        if self.mode == 'mmd':
            if stage == 'test':
                self.dataset_s = MNIST(root=self.mnist_path, train=False, download=True, transform=self.test_transform)
                if self.mmd_image_set_q == 'same_dist':
                    self.dataset_q = MNIST(root=self.mnist_path, train=True, download=True, transform=self.test_transform) 
                elif self.mmd_image_set_q == 'mnist_fake':
                    self.dataset_q = MNIST_FAKE(root=self.mnist_path_fake, transform=self.test_transform) 
                    
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, )
                        #   num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True) 
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, )
                        #   num_workers=8, pin_memory=True, drop_last=True,  persistent_workers=True) 
        
    def test_dataloader(self):
        if self.mode == 'supervised':
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, 
                            pin_memory=True, drop_last=True, shuffle=False, persistent_workers=True) 
        
        if self.mode == 'mmd':
            batch_size = max(self.mmd_sample_sizes)
            num_samples = batch_size * self.mmd_n_tests
            sampler_s = RandomSampler(self.dataset_s, replacement=False, num_samples=3*num_samples)
            sampler_q = RandomSampler(self.dataset_q, replacement=False, num_samples=num_samples)
            dataloader_s = DataLoader(self.dataset_s, batch_size=batch_size*3, sampler=sampler_s)
            dataloader_q = DataLoader(self.dataset_q, batch_size=batch_size, sampler=sampler_q)           
            return CombinedLoader({'s': dataloader_s, 'q': dataloader_q})
        
class HIGGS(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        data_0, data_1 =  pickle.load(open(root, 'rb'))
        
        indices_0 = np.random.choice(data_0.shape[0], size=int(data_0.shape[0] * 0.8), replace=False)
        data_0_train = data_0[indices_0]
        data_0_test = np.delete(data_0, indices_0, axis=0)
        
        indices_1 = np.random.choice(data_1.shape[0], size=int(data_1.shape[0] * 0.8), replace=False)
        data_1_train = data_1[indices_1]
        data_1_test = np.delete(data_1, indices_0, axis=0)
        
        data_train = np.concatenate([data_0_train, data_1_train])
        data_test = np.concatenate([data_0_test, data_1_test])

        self.train_data = data_train[:, 0:-1].astype(np.float32)
        self.train_labels = data_train[:, -1].astype(np.int64)
        self.test_data = data_test[:, 0:-1].astype(np.float32)
        self.test_labels = data_test[:, -1].astype(np.int64)

        scaler = StandardScaler()
        self.train_data = scaler.fit_transform(self.train_data)
        self.test_data = scaler.transform(self.test_data)

        if train:
            self.data = self.train_data
            self.labels = self.train_labels
        else:
            self.data = self.test_data
            self.labels = self.test_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0)      

class RandomCrupt:
    def __init__(self, features_low, features_high, corruption_rate, corruption_level=0.2):
        delta = corruption_level * (torch.Tensor(features_high) - torch.Tensor(features_low))
        self.marginals = Uniform(torch.Tensor(features_low) - delta - 1e-8, 
                                 torch.Tensor(features_high) + delta + 1e-8)
        self.corruption_rate = corruption_rate
        
    def __call__(self, x: Tensor):
        corruption_mask = torch.rand_like(x) > self.corruption_rate
        x_random = self.marginals.sample((1,)).squeeze(0)
        x_corrupted = torch.where(corruption_mask, x_random, x)        
        return x_corrupted

class ContrastiveTransform_V2:
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms
        
    def __call__(self, x):
        return [transform(x) for transform in self.transforms]
        
class HIGGSDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_path = args.data_path
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = HIGGS(self.data_path, train=True)
        self.dataset_test = HIGGS(self.data_path, train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            transforms = [ToTensor(),
                          T.Compose([ToTensor(),
                           RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])]
            self.dataset_train.transform = ContrastiveTransform_V2([ToTensor(),
                          T.Compose([ToTensor(),
                           RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            self.dataset_test.transform = ContrastiveTransform_V2([ToTensor(),
                          T.Compose([ToTensor(),
                           RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          num_workers=8, pin_memory=True, drop_last=True, shuffle=True, persistent_workers=True) 
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, 
                          num_workers=8, pin_memory=True, drop_last=True, shuffle=False, persistent_workers=True) 

class BreastCancer(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        data = datasets.load_breast_cancer(as_frame=True)
        data, target = data["data"], data["target"]
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, stratify=target, random_state=42
        )
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data.to_numpy())
        test_data = scaler.transform(test_data.to_numpy())
        
        if train:
            self.data = train_data.astype(np.float32)
            self.labels = train_target.to_numpy().astype(np.int64)
        else:
            self.data = test_data.astype(np.float32)
            self.labels = test_target.to_numpy().astype(np.int64)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0) 
    
class BreastCancerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = BreastCancer(train=True)
        self.dataset_test = BreastCancer(train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            self.dataset_train.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            self.dataset_test.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            

            # self.dataset_train.transform = ContrastiveTransform(view_transform=
            #                                                     T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)]))
            # self.dataset_test.transform = ContrastiveTransform(view_transform=
            #                                                     T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)]))
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train,  shuffle=True, batch_size=self.batch_size,)
                        #   num_workers=8, pin_memory=True, drop_last=True, shuffle=True, persistent_workers=True) 
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test,  shuffle=False, batch_size=self.batch_size, )
                        #   num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True) 
                        
class AdultIncome(Dataset):
    def __init__(self, data_path, train=True, transform=None):        
        self.train = train
        self.transform = transform
        data, target = self._load_data(data_path)
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, stratify=target, random_state=42
        )
               
        if train:
            self.data = train_data
            self.labels = train_target
        else:
            self.data = test_data
            self.labels = test_target
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0) 
    
    def _load_data(self, data_path):
        df = pd.read_csv(data_path, na_values='?')
        df = df.dropna()
        last_ix = len(df.columns) - 1
        X, y = df.drop(df.columns[last_ix], axis=1), df.iloc[:,last_ix]
        cat_ix = X.select_dtypes(include=['object', 'bool']).columns
        num_ix = X.select_dtypes(include=['int64', 'float64']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_ix),
                ('cat', OneHotEncoder(), cat_ix)
            ])
        X = preprocessor.fit_transform(X).toarray()
        # X = pd.DataFrame(X.toarray())
        # X = X.map(lambda x: np.tanh(x)).to_numpy()
        y = LabelEncoder().fit_transform(y)
        return X.astype(np.float32), y.astype(np.int64)
    
class IncomeDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = AdultIncome(data_path=args.data_path, train=True)
        self.dataset_test = AdultIncome(data_path=args.data_path, train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            self.dataset_train.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            self.dataset_test.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train,  shuffle=True, batch_size=self.batch_size, drop_last=True)
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test,  shuffle=False, batch_size=self.batch_size, drop_last=True)
                        
class GesturePhase(Dataset):
    def __init__(self, data_path, train=True, transform=None):        
        self.train = train
        self.transform = transform
        data, target = self._load_data(data_path)
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, stratify=target, random_state=42
        )
               
        if train:
            self.data = train_data
            self.labels = train_target
        else:
            self.data = test_data
            self.labels = test_target
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0) 
    
    def _load_data(self, data_path):
        data_prefix = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
        data_frames = []
        for prefix in data_prefix:
            df_raw = pd.read_csv(os.path.join(data_path, f"{prefix}_raw.csv") , skiprows=[1,2,3,4])
            df_raw.drop('timestamp',axis=1,inplace=True)
            df_raw.drop('phase',axis=1,inplace=True)
            df_va3 = pd.read_csv(os.path.join(data_path, f"{prefix}_va3.csv"))
            df_va3.rename(columns={'Phase': 'phase'}, inplace=True)
            data_frames.append(pd.concat([df_raw, df_va3],axis=1))
        df= pd.concat(data_frames)
        last_ix = len(df.columns) - 1
        X, y = df.drop(df.columns[last_ix], axis=1), df.iloc[:,last_ix]
        
        # X = X.map(lambda x: np.tanh(x)).to_numpy()
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        return X.astype(np.float32), y.astype(np.int64)
    
class GesturePhaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = GesturePhase(data_path=args.data_path, train=True)
        self.dataset_test = GesturePhase(data_path=args.data_path, train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            self.dataset_train.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([Random_Noise_Mul(), ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            self.dataset_test.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([Random_Noise_Mul(), ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train,  shuffle=True, batch_size=self.batch_size,)
                        #   num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True) 
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test,  shuffle=False, batch_size=self.batch_size, )
                        #   num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True) 
                        
class RobotWall(Dataset):
    def __init__(self, data_path, train=True, transform=None):        
        self.train = train
        self.transform = transform
        data, target = self._load_data(data_path)
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, stratify=target, random_state=42
        )
               
        if train:
            self.data = train_data
            self.labels = train_target
        else:
            self.data = test_data
            self.labels = test_target
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0) 
    
    def _load_data(self, data_path):
        df= pd.read_csv(data_path)
        last_ix = len(df.columns) - 1
        X, y = df.drop(df.columns[last_ix], axis=1), df.iloc[:,last_ix]
        
        X = StandardScaler().fit_transform(X)
        # X = pd.DataFrame(X)       
        # X = X.map(lambda x: np.tanh(x)).to_numpy()
        y = LabelEncoder().fit_transform(y)
        return X.astype(np.float32), y.astype(np.int64)
    
class RobotWallDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = RobotWall(data_path=args.data_path, train=True)
        self.dataset_test = RobotWall(data_path=args.data_path, train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            self.dataset_train.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            self.dataset_test.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train,  shuffle=True, batch_size=self.batch_size, drop_last=True)
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test,  shuffle=False, batch_size=self.batch_size, drop_last=True)
                        
class Theorem(Dataset):
    def __init__(self, data_path, train=True, transform=None):        
        self.train = train
        self.transform = transform

        train_data, test_data, train_target, test_target = self._load_data(data_path)
               
        if train:
            self.data = train_data
            self.labels = train_target
        else:
            self.data = test_data
            self.labels = test_target
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0) 
    
    def _load_data(self, data_path):   
        data_train = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
        data_val = pd.read_csv(os.path.join(data_path, 'validation.csv'), header=None)
        data_test = pd.read_csv(os.path.join(data_path, 'test.csv'), header=None)
        data_train = pd.concat([data_train, data_val])       
        data_train = self._convert_label(data_train)
        data_test = self._convert_label(data_test)
              
        last_ix = len(data_train.columns) - 1       
        train_data, train_target =  data_train.drop(data_train.columns[last_ix], axis=1), data_train.iloc[:,last_ix]      
        test_data, test_target = data_test.drop(data_test.columns[last_ix], axis=1), data_test.iloc[:,last_ix]
        
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data).astype(np.float32)
        test_data = scaler.transform(test_data).astype(np.float32)
        # train_data = train_data.map(lambda x: np.tanh(x)).to_numpy().astype(np.float32)
        # test_data = test_data.map(lambda x: np.tanh(x)).to_numpy().astype(np.float32)

        
        return train_data, test_data, train_target.to_numpy(), test_target.to_numpy()
    
    def _convert_label(self, data_frame):
        label_columns = data_frame.iloc[:, 51:57]
        label_column_index = label_columns.eq(1).idxmax(axis=1) - 51
        df = pd.concat([data_frame, label_column_index], ignore_index=True, axis=1)
        df.drop(label_columns.columns, axis=1, inplace=True)
        return df
    
class TheoremDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = Theorem(data_path=args.data_path, train=True)
        self.dataset_test = Theorem(data_path=args.data_path, train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            transform = T.Compose([ToTensor(), RandomCrupt(-np.ones_like(self.dataset_train.features_low), np.ones_like(self.dataset_train.features_low), 0.6)])
            # self.dataset_train.transform = ContrastiveTransform_V2([
                # ToTensor(),
            #     T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)]),
            #     T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            # self.dataset_test.transform = ContrastiveTransform_V2([
            #     # ToTensor(),
            #     T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)]),
            #     T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            self.dataset_train.transform = ContrastiveTransform(view_transform=transform)
            self.dataset_test.transform = ContrastiveTransform(view_transform=transform)
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train,  shuffle=True, batch_size=self.batch_size, drop_last=True) 
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test,  shuffle=False, batch_size=self.batch_size, drop_last=True)
    

class Obesity(Dataset):
    def __init__(self, data_path, train=True, transform=None):        
        self.train = train
        self.transform = transform
        data, target = self._load_data(data_path)
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, stratify=target, random_state=42
        )
               
        if train:
            self.data = train_data
            self.labels = train_target
        else:
            self.data = test_data
            self.labels = test_target
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0) 
    
    def _load_data(self, data_path):
        df = pd.read_csv(data_path, na_values='?')
        df = df.dropna()
        last_ix = len(df.columns) - 1
        X, y = df.drop(df.columns[last_ix], axis=1), df.iloc[:,last_ix]
        cat_ix = X.select_dtypes(include=['object', 'bool']).columns
        num_ix = X.select_dtypes(include=['int64', 'float64']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_ix),
                ('cat', OneHotEncoder(), cat_ix)
            ])
        X = preprocessor.fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        return X.astype(np.float32), y.astype(np.int64)
    
class ObesityDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = Obesity(data_path=args.data_path, train=True)
        self.dataset_test = Obesity(data_path=args.data_path, train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            self.dataset_train.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            self.dataset_test.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train,  shuffle=True, batch_size=self.batch_size, drop_last=True) 
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test,  shuffle=False, batch_size=self.batch_size, drop_last=True)
    
class Ozone(Dataset):
    def __init__(self, data_path, train=True, transform=None):        
        self.train = train
        self.transform = transform
        data, target = self._load_data(data_path)
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, stratify=target, random_state=42
        )
               
        if train:
            self.data = train_data
            self.labels = train_target
        else:
            self.data = test_data
            self.labels = test_target
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0) 
    
    def _load_data(self, data_path):
        df = pd.read_csv(data_path, na_values='?')
        df = df.dropna()
        last_ix = len(df.columns) - 1
        X, y = df.drop(df.columns[last_ix], axis=1), df.iloc[:,last_ix]
        cat_ix = X.select_dtypes(include=['object', 'bool']).columns
        num_ix = X.select_dtypes(include=['int64', 'float64']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_ix),
                ('cat', OneHotEncoder(), cat_ix)
            ])
        X = preprocessor.fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        return X.astype(np.float32), y.astype(np.int64)
    
class OzoneModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = Ozone(data_path=args.data_path, train=True)
        self.dataset_test = Ozone(data_path=args.data_path, train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            self.dataset_train.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            self.dataset_test.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train,  shuffle=True, batch_size=self.batch_size, drop_last=True) 
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test,  shuffle=False, batch_size=self.batch_size, drop_last=True)
    
class Texture(Dataset):
    def __init__(self, data_path, train=True, transform=None):        
        self.train = train
        self.transform = transform
        data, target = self._load_data(data_path)
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, stratify=target, random_state=42
        )
               
        if train:
            self.data = train_data
            self.labels = train_target
        else:
            self.data = test_data
            self.labels = test_target
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0) 
    
    def _load_data(self, data_path):
        data, _ = arff.loadarff(data_path)   
        df = pd.DataFrame(data)
        df = df.dropna()
        last_ix = len(df.columns) - 1
        X, y = df.drop(df.columns[last_ix], axis=1), df.iloc[:,last_ix]
        cat_ix = X.select_dtypes(include=['object', 'bool']).columns
        num_ix = X.select_dtypes(include=['int64', 'float64']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_ix),
                ('cat', OneHotEncoder(), cat_ix)
            ])
        X = preprocessor.fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        return X.astype(np.float32), y.astype(np.int64)
    
class TextureModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = Texture(data_path=args.data_path, train=True)
        self.dataset_test = Texture(data_path=args.data_path, train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            self.dataset_train.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            self.dataset_test.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train,  shuffle=True, batch_size=self.batch_size, drop_last=True) 
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test,  shuffle=False, batch_size=self.batch_size, drop_last=True)
    
class DNA(Dataset):
    def __init__(self, data_path, train=True, transform=None):        
        self.train = train
        self.transform = transform
        data, target = self._load_data(data_path)
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, stratify=target, random_state=42
        )
               
        if train:
            self.data = train_data
            self.labels = train_target
        else:
            self.data = test_data
            self.labels = test_target
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]  
    
    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0) 
    
    def _load_data(self, data_path):
        data, _ = arff.loadarff(data_path)   
        df = pd.DataFrame(data)
        df = df.dropna()
        last_ix = len(df.columns) - 1
        X, y = df.drop(df.columns[last_ix], axis=1), df.iloc[:,last_ix]
        cat_ix = X.select_dtypes(include=['object', 'bool']).columns
        num_ix = X.select_dtypes(include=['int64', 'float64']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_ix),
                ('cat', OneHotEncoder(), cat_ix)
            ])
        X = preprocessor.fit_transform(X)
        # X = pd.DataFrame(X)       
        # X = X.map(lambda x: np.tanh(x)).to_numpy()
        y = LabelEncoder().fit_transform(y)
        return X.astype(np.float32), y.astype(np.int64)
    
class DNAModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        assert self.mode in ['unsupervised', 'supervised']
        self.dataset_train = DNA(data_path=args.data_path, train=True)
        self.dataset_test = DNA(data_path=args.data_path, train=False)
        self.batch_size = args.batch_size
        
    def setup(self, stage: str):
        if self.mode == 'unsupervised':
            self.dataset_train.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            self.dataset_test.transform = ContrastiveTransform_V2([
                ToTensor(),
                T.Compose([ToTensor(), RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            
            # self.dataset_train.transform = ContrastiveTransform_V2([
            #     ToTensor(),
            #     T.Compose([Random_Noise_Mul(), Random_Mask(p=0.2), ToTensor(), 
            #                RandomCrupt(self.dataset_train.features_low, self.dataset_train.features_high, 0.6)])])
            
            # self.dataset_test.transform = ContrastiveTransform_V2([
            #     ToTensor(),
            #     T.Compose([Random_Noise_Mul(), Random_Mask(p=0.2), ToTensor(), 
            #                RandomCrupt(self.dataset_test.features_low, self.dataset_test.features_high, 0.6)])])
            
        elif self.mode == 'supervised':
            self.dataset_train.transform = ToTensor()
            self.dataset_test.transform = ToTensor()
                           
    def train_dataloader(self): 
        return DataLoader(self.dataset_train,  shuffle=True, batch_size=self.batch_size, drop_last=True) 
        
    def val_dataloader(self):
        return DataLoader(self.dataset_test,  shuffle=False, batch_size=self.batch_size, drop_last=True)