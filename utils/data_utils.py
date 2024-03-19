from typing import Tuple, Union, List, Optional
from PIL import Image
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, RandomSampler, BatchSampler
from torchvision.datasets import CIFAR10, ImageFolder
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

class CADetTransform:
    def __init__(
        self,
        num_tranforms: int = 50,
        input_size: int = 224,
        hf_prob: float = 0.5
    ):
        # view_transform = T.Compose([
        #     T.RandomResizedCrop(size=input_size, scale=(0.75, 0.75)),
        #     # T.RandomHorizontalFlip(p=hf_prob),
        #     T.ToTensor(),
        #     T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
        # ])
        view_transform = SimCLRViewTransform()
        self.transforms = [view_transform for _ in range(num_tranforms)]

    def __call__(self, image):
         return [transform(image) for transform in self.transforms]
    
class CADetTransform_CIFAR:
    def __init__(
        self,
        num_tranforms: int = 50,
        hf_prob: float = 0.5
    ):
        # view_transform = T.Compose([
        #     T.RandomResizedCrop(size=32, scale=(0.75, 0.75)),
        #     T.RandomHorizontalFlip(p=hf_prob),
        #     T.ToTensor(),
        #     T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
        # ])
        view_transform = SimCLRViewTransform(input_size=32, min_scale=0.75, cj_prob=0)
        self.transforms = [view_transform for _ in range(num_tranforms)]

    def __call__(self, image):
         return [transform(image) for transform in self.transforms]
    
class PixelFlick:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image):
        image = np.array(image, dtype=np.uint8)
        noise = torch.randint(-self.scale, self.scale, image.shape)
        image =  np.clip(image + noise.numpy(), 0, 255).astype(np.uint8)
        return Image.fromarray(image)

class CIFAR10_NPY(Dataset):
    """ load cifar dataset from .npy format
    """
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data = np.load(root, allow_pickle=True)

    def __len__(self):
        if self.data.dtype != 'uint8':
            # dict
            return len(self.data.item()['data'])
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.data.dtype != 'uint8':
            # dict
            sample = self.data.item()['data'][index]
            label = self.data.item()['labels'][index]
        else:
            sample = self.data[index]
            label = 10 

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
                # self.dataset_s = CIFAR10(root=self.cifar10_path, train=False, download=True, transform=self.test_transform)
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
                self.test_dataset_same_dist = CIFAR10(root=self.cifar10_path, train=False, download=True, transform=transform) 
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
            sampler_s = RandomSampler(self.dataset_s, replacement=True, num_samples=3*num_samples)
            sampler_q = RandomSampler(self.dataset_q, replacement=True, num_samples=num_samples)
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
            sampler_test_same_dist = RandomSampler(self.test_dataset_same_dist, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_cifar10_1 = RandomSampler(self.test_dataset_cifar10_1, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_pgd = RandomSampler(self.test_dataset_pgd, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_cw = RandomSampler(self.test_dataset_cw, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_fgsm = RandomSampler(self.test_dataset_fgsm, replacement=False, num_samples=self.cadet_n_tests)
            
            dataloader_test_same_dist = DataLoader(self.test_dataset_same_dist, batch_size=1, sampler=sampler_test_same_dist, collate_fn=self.cadet_collate_fn)
            dataloader_test_cifar10_1 = DataLoader(self.test_dataset_cifar10_1, batch_size=1, sampler=sampler_test_cifar10_1, collate_fn=self.cadet_collate_fn)
            dataloader_test_pgd = DataLoader(self.test_dataset_pgd, batch_size=1, sampler=sampler_test_pgd, collate_fn=self.cadet_collate_fn)
            dataloader_test_cw = DataLoader(self.test_dataset_cw, batch_size=1, sampler=sampler_test_cw, collate_fn=self.cadet_collate_fn)
            dataloader_test_fgsm = DataLoader(self.test_dataset_fgsm, batch_size=1, sampler=sampler_test_fgsm, collate_fn=self.cadet_collate_fn)
            
            return CombinedLoader({'same_dist': dataloader_test_same_dist,  'cifar10_1': dataloader_test_cifar10_1,
                                   'pgd': dataloader_test_pgd, 'cw': dataloader_test_cw, 'fgsm': dataloader_test_fgsm})

    @staticmethod
    def cadet_collate_fn(batch):
        X, _ = batch[0]
        return torch.stack(X)
    
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
        assert self.mmd_image_set_q in ['same_dist', 'imagenet_o', 'inaturalist', 'pgd', 'cw', 'fgsm']

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
                transform = CADetTransform(num_tranforms=self.cadet_n_transforms)
                # self.test_dataset_same_dist = ImageFolder(root=self.train_path, transform=transform)
                self.test_dataset_same_dist = ImageFolder(root=self.val_path, transform=transform)
                self.test_dataset_imagenet_o = ImageFolder(root=self.imagenet_o_path, transform=transform)
                self.test_dataset_inaturalist = ImageFolder(root=self.inaturalist_path, transform=transform)
                self.test_dataset_pgd = ImageFolder(root=self.pgd_path, transform=transform)
                self.test_dataset_cw = ImageFolder(root=self.cw_path, transform=transform)
                self.test_dataset_fgsm = ImageFolder(root=self.fgsm_path, transform=transform)
 
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
            dataloader_s = DataLoader(self.dataset_s, batch_size=batch_size*3, sampler=sampler_s, )
                                    # num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
            dataloader_q = DataLoader(self.dataset_q, batch_size=1, sampler=sampler_q, collate_fn=self.mmd_ss_collate_fn)
                                    # num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
            
            return CombinedLoader({'s': dataloader_s, 'q': dataloader_q})
        
        if self.mode == 'cadet':
            sampler_test_same_dist = RandomSampler(self.test_dataset_same_dist, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_imagenet_o = RandomSampler(self.test_dataset_imagenet_o, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_inaturalist = RandomSampler(self.test_dataset_inaturalist, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_pgd = RandomSampler(self.test_dataset_pgd, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_cw = RandomSampler(self.test_dataset_cw, replacement=False, num_samples=self.cadet_n_tests)
            sampler_test_fgsm = RandomSampler(self.test_dataset_fgsm, replacement=False, num_samples=self.cadet_n_tests)
            dataloader_test_same_dist = DataLoader(self.test_dataset_same_dist, batch_size=1, sampler=sampler_test_same_dist, collate_fn=self.cadet_collate_fn)
            dataloader_test_imagenet_o = DataLoader(self.test_dataset_imagenet_o, batch_size=1, sampler=sampler_test_imagenet_o, collate_fn=self.cadet_collate_fn)
            dataloader_test_inaturalist = DataLoader(self.test_dataset_inaturalist, batch_size=1, sampler=sampler_test_inaturalist, collate_fn=self.cadet_collate_fn)
            dataloader_test_pgd = DataLoader(self.test_dataset_pgd, batch_size=1, sampler=sampler_test_pgd, collate_fn=self.cadet_collate_fn)
            dataloader_test_cw = DataLoader(self.test_dataset_cw, batch_size=1, sampler=sampler_test_cw, collate_fn=self.cadet_collate_fn)
            dataloader_test_fgsm = DataLoader(self.test_dataset_fgsm, batch_size=1, sampler=sampler_test_fgsm, collate_fn=self.cadet_collate_fn)
            return CombinedLoader({'same_dist': dataloader_test_same_dist, 'imagenet_o': dataloader_test_imagenet_o, 'inaturalist': dataloader_test_inaturalist, 
                                   'pgd': dataloader_test_pgd, 'cw': dataloader_test_cw, 'fgsm': dataloader_test_fgsm})

    @staticmethod
    def cadet_collate_fn(batch):
        X, _ = batch[0]
        return torch.stack(X)
    
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
    
    