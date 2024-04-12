from utils.data_utils import (
    CIFAR10_NPY, CIFAR10, 
    CADetTransform_CIFAR, 
    Cifar10_ByClass, 
    MNIST_FAKE, 
    Convert, 
    ContrastiveTransform,
    Random_Mask,
    ToTensor,
    L2_Normalize,
    HIGGS)
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder, MNIST
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import torchvision.transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
from lightly.models.utils import deactivate_requires_grad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lightly.transforms import SimCLRTransform, SimCLRViewTransform
from utils.stat_utils import draw_roc_curve, mmd_permutation_test, get_rejection_rate
from sklearn.metrics import roc_curve, auc

def cal_similarity(input0, input1):
    # return torch.exp(-torch.cdist(input0, input1, p=2))
    out0 = F.normalize(input0, dim=1)
    out1 = F.normalize(input1, dim=1)
    return out0 @ out1.t()

def calculate_features():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone    
    transform = T.Compose([
        T.ToTensor(), 
        T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
    ])
    # full_dataset = Cifar10_ByClass(CIFAR10('image_data/Cifar10', train=False, 
    #                                        transform=transform))   
    # full_dataset = Cifar10_ByClass(CIFAR10_NPY('image_data/Cifar10/cifar10_val_PGD.npy',  
    #                                        transform=transform))   
    # full_dataset = Cifar10_ByClass(CIFAR10_NPY('image_data/Cifar10/cifar10_val_CW.npy',  
    #                                        transform=transform))   
    full_dataset = Cifar10_ByClass(CIFAR10_NPY('image_data/Cifar10/cifar10_val_FGSM.npy',  
                                           transform=transform))   
 
    features = []
    with torch.inference_mode():
        for i in range(10):
            feats = []
            subset = full_dataset.get_dataset_by_class_id(i)
            dataloader = DataLoader(subset, batch_size=128)
            for x, _ in tqdm(dataloader, desc=f'Calculate features for class {i}'):
                feat = backbone(x.to(device))
                feats.append(feat)
            feats = torch.cat(feats, dim=0)
            features.append(feats.cpu().numpy())
    os.makedirs('Features', exist_ok=True)
    np.save('Features/cifar10_val_fgsm.npy', features)
 

def intra_sim_matrix_cifar10(path):
    features = np.load(path)
    sim_mat = torch.zeros(10, 10)
    for i in range(10):
        for j in range(10):
            similarity = cal_similarity(torch.tensor(features[i]),
                                        torch.tensor(features[j]))
            if i == j:
                n = features[i].shape[0]
                sim_mat[i,j] = (similarity.sum() - similarity.trace()) / (n * (n - 1))
            else:
                sim_mat[i,j] = similarity.mean()

    df = pd.DataFrame(sim_mat)
    plt.figure(figsize=(8, 8))  
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
    plt.title("In Distribution Similarities")
    plt.show()            

    return sim_mat

def inter_sim_matrix_cifar10(id_path, od_path):
    sim_mat = torch.zeros(10, 10)
    features_in = np.load(id_path)
    features_od = np.load(od_path)
    for i in range(10):
        for j in range(10):
            similarity = cal_similarity(torch.tensor(features_in[i]),
                                        torch.tensor(features_od[j]))
            sim_mat[i,j] = similarity.mean()
            
    df = pd.DataFrame(sim_mat)
    plt.figure(figsize=(8, 8))  
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
    plt.title("In/Out Distribution Similarities")
    plt.show()   
    
    return sim_mat

def similarity_aug():
    class _transform:
        def __init__(self, num_tranforms: int):
            self.view_transform = SimCLRViewTransform(input_size=32)
            # self.view_transform = T.Compose([Convert('RGB'), SimCLRViewTransform(input_size=32, cj_prob=0)])
            # self.view_transform = T.Compose([
            #     T.Resize([32,32], interpolation=T.InterpolationMode.BILINEAR), 
            #     T.RandomResizedCrop(size=32, scale=(0.75, 0.75)),
            #     T.RandomHorizontalFlip(),
            #     T.ToTensor(),
            #     T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
            # ])
            self.n_transforms = num_tranforms
            
        def __call__(self, image):
            return torch.stack([self.view_transform(image) for _ in range(self.n_transforms)])
        
    cfg = OmegaConf.load('configs/config_cifar.yml')
    # cfg = OmegaConf.load('configs/config_mnist.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone  
    n_trans = 50
    sample_size = 1000
    transform = _transform(num_tranforms=n_trans)
    dataset_id = CIFAR10('image_data/Cifar10', train=False, transform=transform)
    # dataset_od = CIFAR10_NPY('image_data/Cifar10/cifar10_val_FGSM.npy', transform=transform)
    # dataset_od = CIFAR10_NPY('image_data/Cifar10/cifar10_val_PGD.npy', transform=transform)
    # dataset_od = CIFAR10_NPY('image_data/Cifar10/cifar10_val_CW.npy', transform=transform)
    dataset_od = CIFAR10_NPY('image_data/Cifar10_1/cifar10.1_v4_data.npy', transform=transform)
    dataset_id, _ = random_split(dataset_id, [sample_size, len(dataset_id)-sample_size])
    dataset_od, _ = random_split(dataset_od, [sample_size, len(dataset_od)-sample_size])
    
    # dataset_id, dataset_od, _ = random_split(dataset_id, [sample_size, sample_size, len(dataset_id)-2*sample_size])
    
    # dataset_id = MNIST('image_data/MNIST', train=False, transform=transform)
    # dataset_od = MNIST_FAKE('image_data/MNIST/Fake_MNIST_data_EP100_N10000.pckl', transform=transform)
    # dataset_id, _ = random_split(dataset_id, [sample_size, len(dataset_id)-sample_size])
    # dataset_od, _ = random_split(dataset_od, [sample_size, len(dataset_od)-sample_size])
    
    mask = ~torch.eye(n_trans, dtype=bool)
    
    dataloader_id = DataLoader(dataset_id, batch_size=1)   
    sims_id = []
    for X, _ in tqdm(dataloader_id, desc='calculate in distribution'):
        feat = backbone(X[0].to(device))
        sim = cal_similarity(feat, feat)[mask].mean()
        sims_id.append(sim.item())       
    sims_id = torch.tensor(sims_id)
    
    dataloader_od = DataLoader(dataset_od, batch_size=1)   
    sims_od = []
    for X, _ in tqdm(dataloader_od, desc='calculate out distribution'):
        feat = backbone(X[0].to(device))
        sim = cal_similarity(feat, feat)[mask].mean()
        sims_od.append(sim.item())        
    sims_od = torch.tensor(sims_od)
    
    print(f'similarity id: min: {sims_id.min()}, mean: {sims_id.mean()}, max: {sims_id.max()}, std: {sims_id.std()}')
    print(f'similarity od: min: {sims_od.min()}, mean: {sims_od.mean()}, max: {sims_od.max()}, std: {sims_od.std()}')
    print(f'id - od = {sims_id.mean() - sims_od.mean()}')
    sns.kdeplot(sims_id, color='red', linewidth=2, label='sim_id')
    sns.kdeplot(sims_od, color='blue', linewidth=2, label='sim_od')
    # kde_plot = sns.kdeplot(sims_id - sims_od, color='blue', linewidth=2, label='sim_id - sim_od')
    # kde_data = kde_plot.get_lines()[0].get_data()
    # peak_index = np.argmax(kde_data[1]) 
    # peak_value = kde_data[0][peak_index]
    # plt.axvline(peak_value, color='red', linestyle='--', label=f'mode={peak_value:.4f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title("Similarity across augmentations")
    plt.legend(loc="upper right")
    plt.show()
    
def similarity_cross():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    # cfg = OmegaConf.load('configs/config_mnist.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone  
    
    # transform = T.Compose([T.ToTensor(), 
                        #    T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])])
    transform = SimCLRViewTransform(input_size=32)               
    # transform = T.Compose([Convert('RGB'), SimCLRViewTransform(input_size=32, cj_prob=0)])
    holdout_size = 5000
    sample_size = 2000
    dataset_id = CIFAR10('image_data/Cifar10', train=False, transform=transform)
    # dataset_od = CIFAR10_NPY('image_data/Cifar10/cifar10_val_FGSM.npy', transform=transform)
    # dataset_od = CIFAR10_NPY('image_data/Cifar10/cifar10_val_PGD.npy', transform=transform)
    # dataset_od = CIFAR10_NPY('image_data/Cifar10/cifar10_val_CW.npy', transform=transform)
    dataset_od = CIFAR10_NPY('image_data/Cifar10_1/cifar10.1_v4_data.npy', transform=transform)
    dataset_holdout, dataset_id, _ = random_split(dataset_id, [holdout_size, sample_size, 
                                                               len(dataset_id)-holdout_size-sample_size])
    dataset_od, _ = random_split(dataset_od, [sample_size, len(dataset_od)-sample_size])
    # dataset_holdout, dataset_id, dataset_od, _ = random_split(dataset_id, 
    #                             [holdout_size, sample_size, sample_size, len(dataset_id)-holdout_size-2*sample_size])
    
    # dataset_id = MNIST('image_data/MNIST', train=False, transform=transform)
    # dataset_holdout, dataset_id, _ = random_split(dataset_id, [holdout_size, sample_size, 
    #                                                            len(dataset_id)-holdout_size-sample_size])
    # dataset_od = MNIST_FAKE('image_data/MNIST/Fake_MNIST_data_EP100_N10000.pckl', transform=transform)
    
    feats_holdout = []
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=128)   
    feats_holdout = []
    for X, _ in tqdm(dataloader_holdout, desc='calculate hold out'):
        feat = backbone(X.to(device))
        feats_holdout.append(feat)        
    feats_holdout = torch.cat(feats_holdout)   
    
    dataloader_id = DataLoader(dataset_id, batch_size=128)   
    feats_id = []
    for X, _ in tqdm(dataloader_id, desc='calculate in distribution'):
        feat = backbone(X.to(device))
        feats_id.append(feat)        
    feats_id = torch.cat(feats_id)   
    sims_id = cal_similarity(feats_id, feats_holdout).mean(dim=1)
    
    dataloader_od = DataLoader(dataset_od, batch_size=128)   
    feats_od = []
    for X, _ in tqdm(dataloader_od, desc='calculate out distribution'):
        feat = backbone(X.to(device))
        feats_od.append(feat)        
    feats_od = torch.cat(feats_od)    
    sims_od = cal_similarity(feats_od, feats_holdout).mean(dim=1)
        
    print(f'similarity id: min: {sims_id.min()}, mean: {sims_id.mean()}, max: {sims_id.max()}, std: {sims_id.std()}')
    print(f'similarity od: min: {sims_od.min()}, mean: {sims_od.mean()}, max: {sims_od.max()}, std: {sims_od.std()}')
    
    # kde_plot = sns.kdeplot((sims_od - sims_id).cpu(), color='blue', linewidth=2, label='sim_od - sim_id')
    # kde_data = kde_plot.get_lines()[0].get_data()
    # peak_index = np.argmax(kde_data[1]) 
    # peak_value = kde_data[0][peak_index]
    # plt.axvline(peak_value, color='red', linestyle='--', label=f'mode={peak_value:.4f}')
    mean_diff = sims_od.mean() - sims_id.mean()
    # plt.axvline(mean_diff.item(), color='red', linestyle='--', 
    #             label=f'mean={mean_diff.item():.4f}')
    
    print(f'od - id = {mean_diff}')
    sns.kdeplot(sims_id.cpu(), color='red', linewidth=2, label='sim_id')
    sns.kdeplot(sims_od.cpu(), color='blue', linewidth=2, label='sim_od')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title("Similarities across samples")
    plt.legend(loc="upper right")
    plt.show()
    
def similarity_test():
    id_path = 'Features/cifar10_val.npy'
    od_path = 'Features/cifar10_val_fgsm.npy'
    intra_sim = intra_sim_matrix_cifar10(id_path)
    inter_sim = inter_sim_matrix_cifar10(id_path, od_path)
    diff_sim = intra_sim - inter_sim
    df = pd.DataFrame(diff_sim)
    plt.figure(figsize=(8, 8))  
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
    plt.title("Similarities Differences")
    plt.show()   
    
    return diff_sim

def contrastive_two_sample_test(sample_size_test):
    view_transform = SimCLRViewTransform(input_size=32)
    class _transform:
        def __init__(self, num_tranforms: int):
            self.view_transform = view_transform
            self.n_transforms = num_tranforms
            
        def __call__(self, image):
            return torch.stack([self.view_transform(image) for _ in range(self.n_transforms)])
    
    cfg = OmegaConf.load('configs/config_cifar.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
    
    n_holdout = 5000
    n_trans = 50
    sample_size_calib = 500
    n_test = 100
    mask = ~torch.eye(n_trans, dtype=bool)
    dataset_full = CIFAR10('image_data/Cifar10', train=False, transform=view_transform)
    dataset_od = CIFAR10_NPY('image_data/Cifar10/cifar10_val_CW.npy', transform=view_transform)
    # dataset_od = CIFAR10_NPY('image_data/Cifar10/cifar10_val_PGD.npy', transform=view_transform)
    # dataset_od = CIFAR10_NPY('image_data/Cifar10/cifar10_val_FGSM.npy', transform=view_transform)
    # dataset_od = CIFAR10_NPY('image_data/Cifar10_1/cifar10.1_v4_data.npy', transform=view_transform)
    # dataset_od = ImageFolder('//wsl.localhost/Ubuntu/home/bingbing/codes/OpenOOD/data/images_classic/cifar100/test',  transform=view_transform)
    dataset_holdout, dataset_calib, dataset_id = random_split(dataset_full, [n_holdout, sample_size_calib, len(dataset_full)-n_holdout-sample_size_calib])
    
    # calculate holdout features
    feats_holdout = []
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=128)   
    for X, _ in tqdm(dataloader_holdout, desc='calculate hold out features'):
        feat = backbone(X.to(device))
        feats_holdout.append(feat)        
    feats_holdout = torch.cat(feats_holdout)   
    
    # calibrate 
    dataloader_calib = DataLoader(dataset_calib, batch_size=128)
    feats_calib = []
    for X, _ in tqdm(dataloader_calib, desc='calibrate in distribution cross-sim'):
        feat = backbone(X.to(device))
        feats_calib.append(feat)        
    feats_calib= torch.cat(feats_calib)   
    cross_sim_calib = cal_similarity(feats_calib, feats_holdout).mean()
    print(f'in distribution cross-sim: {cross_sim_calib.item()}')
    
    
    # dataset_calib.dataset.transform = _transform(num_tranforms=n_trans)
    sims = []
    for X, _ in tqdm(dataset_calib, desc='calibrate in distribution aug-sim'):
        feat = backbone(X.to(device))
        sim = cal_similarity(feat, feat)[mask].mean()
        sims.append(sim.item())       
    aug_sims_calib = torch.tensor(sims).mean()
    print(f'in distribution aug-sim: {aug_sims_calib.item()}')
    
    stat_calib = cross_sim_calib - aug_sims_calib
      
    stats_od = []
    stats_id = []
    for i in tqdm(range(n_test), desc='two sample test'):
        dataset_od_test, _ = random_split(dataset_od, [sample_size_test, len(dataset_od)-sample_size_test])
        dataset_id_test, _ = random_split(dataset_id, [sample_size_test, len(dataset_id)-sample_size_test])
        
        # print('calculate aug_sim_diff')
        dataset_od_test.dataset.transform = _transform(num_tranforms=n_trans)
        dataset_id_test.dataset.dataset.transform = _transform(num_tranforms=n_trans)
        
        sims = []
        for X, _ in dataset_od_test:
            feat = backbone(X.to(device))
            sim = cal_similarity(feat, feat)[mask].mean()
            sims.append(sim.item())       
        aug_sims_od = torch.tensor(sims).mean()
        
        sims = []
        for X, _ in dataset_id_test:
            feat = backbone(X.to(device))
            sim = cal_similarity(feat, feat)[mask].mean()
            sims.append(sim.item())       
        aug_sims_id = torch.tensor(sims).mean()
        
        # print('calculate cross sample similarities')
        dataset_od_test.dataset.transform = view_transform
        dataset_id_test.dataset.dataset.transform = view_transform
        dataloader_od_test = DataLoader(dataset_od_test, batch_size=128)   
        dataloader_id_test = DataLoader(dataset_id_test, batch_size=128)
        
        feats_od = []
        for X, _ in dataloader_od_test:
            feat = backbone(X.to(device))
            feats_od.append(feat)        
        feats_od = torch.cat(feats_od)   
        cross_sims_od = cal_similarity(feats_od, feats_holdout).mean()
             
        feats_id = []
        for X, _ in dataloader_id_test:
            feat = backbone(X.to(device))
            feats_id.append(feat)        
        feats_id = torch.cat(feats_id)   
        cross_sims_id = cal_similarity(feats_id, feats_holdout).mean()
        
        # stat_od = (aug_sims_calib - aug_sims_od) + (cross_sims_od - cross_sim_calib)
        # stat_id = (aug_sims_calib - aug_sims_id) + (cross_sims_id - cross_sim_calib)
        stat_od = - aug_sims_od + cross_sims_od
        stat_id = - aug_sims_id + cross_sims_id
        stats_od.append(stat_od.item())
        stats_id.append(stat_id.item())
        # print(f'stat_id: {stat_id.item()}\tstat_od: {stat_od.item()}')
    result = {
        'stats_od': np.array(stats_od),
        'stats_id': np.array(stats_id)
    }
    np.save(f'two_sample_test_results/contrastive_simclr.npy', result)
    print(f'rejection rate: {get_rejection_rate(stats_id, stats_od)}')
    draw_roc_curve(stats_id, stats_od)


def mmd_two_sample_test(sample_size):
    view_transform = SimCLRViewTransform(input_size=32)
    cfg = OmegaConf.load('configs/config_cifar.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
    
    n_test = 100
    dataset_full = CIFAR10('image_data/Cifar10', train=False, transform=view_transform)
    # dataset_od_full = CIFAR10_NPY('image_data/Cifar10/cifar10_val_CW.npy', transform=view_transform)
    # dataset_od_full = CIFAR10_NPY('image_data/Cifar10/cifar10_val_PGD.npy', transform=view_transform)
    # dataset_od_full = CIFAR10_NPY('image_data/Cifar10/cifar10_val_FGSM.npy', transform=view_transform)
    dataset_od_full = CIFAR10_NPY('image_data/Cifar10_1/cifar10.1_v4_data.npy', transform=view_transform)
    # dataset_od_full = ImageFolder('//wsl.localhost/Ubuntu/home/bingbing/codes/OpenOOD/data/images_classic/cifar100/test',  transform=view_transform)
    dataset_holdout, dataset_id_full = random_split(dataset_full, [sample_size, len(dataset_full)-sample_size])
    
    
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=sample_size)
    feats_holdout = []
    for X_holdout, _ in dataloader_holdout:
        feat_holdout = backbone(X_holdout.to(device))
        feats_holdout.append(feat_holdout)
    feats_holdout = torch.cat(feats_holdout)
    
    mmds_id = []
    mmds_od = []
    for i in tqdm(range(n_test), desc='two sample test'):
        dataset_od, _ = random_split(dataset_od_full, [sample_size, len(dataset_od_full)-sample_size])
        dataset_id, _ = random_split(dataset_id_full, [sample_size, len(dataset_id_full)-sample_size])
        dataloader_od = DataLoader(dataset_od, batch_size=sample_size)   
        dataloader_id = DataLoader(dataset_id, batch_size=sample_size)
        feats_od = []
        feats_id = []
        for (X_od, _), (X_id, _) in zip(dataloader_od, dataloader_id):
            feat_od = backbone(X_od.to(device))
            feat_id = backbone(X_id.to(device))
            feats_od.append(feat_od)
            feats_id.append(feat_id)
        feats_od = torch.cat(feats_od)
        feats_id = torch.cat(feats_id)
        
        _, _, mmd_id = mmd_permutation_test(feats_holdout, feats_id)
        _, _, mmd_od = mmd_permutation_test(feats_holdout, feats_od)  
        mmds_id.append(mmd_id)
        mmds_od.append(mmd_od)
    result = {
        'stats_od': np.array(mmds_od),
        'stats_id': np.array(mmds_id)
    }
    np.save(f'two_sample_test_results/mmd_simclr_{sample_size}.npy', result)
    print(f'rejection rate: {get_rejection_rate(mmds_id, mmds_od)}')
    draw_roc_curve(mmds_id, mmds_od)
    
def mmd_two_sample_test_mnist():
    view_transform = T.Compose([Convert('RGB'), SimCLRViewTransform(input_size=32, cj_prob=0)])
    cfg = OmegaConf.load('configs/config_mnist.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
    
    sample_size = 10
    n_test = 100
    dataset_full = MNIST('image_data/MNIST', train=False, transform=view_transform)
    dataset_od_full = MNIST_FAKE('image_data/MNIST/Fake_MNIST_data_EP100_N10000.pckl', transform=view_transform)
    dataset_holdout, dataset_id_full = random_split(dataset_full, [sample_size, len(dataset_full)-sample_size])
    
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=sample_size)
    feats_holdout = []
    for X_holdout, _ in dataloader_holdout:
        feat_holdout = backbone(X_holdout.to(device))
        feats_holdout.append(feat_holdout)
    feats_holdout = torch.cat(feats_holdout)
    
    mmds_id = []
    mmds_od = []
    for i in tqdm(range(n_test), desc='two sample test'):
        dataset_od, _ = random_split(dataset_od_full, [sample_size, len(dataset_od_full)-sample_size])
        dataset_id, _ = random_split(dataset_id_full, [sample_size, len(dataset_id_full)-sample_size])
        dataloader_od = DataLoader(dataset_od, batch_size=sample_size)   
        dataloader_id = DataLoader(dataset_id, batch_size=sample_size)
        feats_od = []
        feats_id = []
        for (X_od, _), (X_id, _) in zip(dataloader_od, dataloader_id):
            feat_od = backbone(X_od.to(device))
            feat_id = backbone(X_id.to(device))
            feats_od.append(feat_od)
            feats_id.append(feat_id)
        feats_od = torch.cat(feats_od)
        feats_id = torch.cat(feats_id)
        
        _, _, mmd_id = mmd_permutation_test(feats_holdout, feats_id)
        _, _, mmd_od = mmd_permutation_test(feats_holdout, feats_od)  
        mmds_id.append(mmd_id)
        mmds_od.append(mmd_od)
    print(f'rejection rate: {get_rejection_rate(mmds_id, mmds_od)}')
    draw_roc_curve(mmds_id, mmds_od)
    
def cadet_test():
    class _transform:
        def __init__(self, num_tranforms: int):
            self.view_transform = SimCLRViewTransform(input_size=32)
            # self.view_transform = T.Compose([
            #     T.Resize([32,32], interpolation=T.InterpolationMode.BILINEAR),
            #     T.RandomResizedCrop(size=32, scale=(0.75, 0.75)),
            #     T.RandomHorizontalFlip(),
            #     T.ToTensor(),
            #     T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
            # ])
            self.n_transforms = num_tranforms
            
        def __call__(self, image):
            return torch.stack([self.view_transform(image) for _ in range(self.n_transforms)])
    
    cfg = OmegaConf.load('configs/config_cifar.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
    
    n_val1 = 300
    n_val2 = 2000
    n_trans = 50
    transform = _transform(num_tranforms=n_trans)
    n_test = 100
    mask = ~torch.eye(n_trans, dtype=bool)
    dataset_full = CIFAR10('image_data/Cifar10', train=False, transform=transform)
    # dataset_od_full = CIFAR10_NPY('image_data/Cifar10/cifar10_val_CW.npy', transform=transform)
    # dataset_od_full = CIFAR10_NPY('image_data/Cifar10/cifar10_val_PGD.npy', transform=transform)
    # dataset_od_full = CIFAR10_NPY('image_data/Cifar10/cifar10_val_FGSM.npy', transform=transform)
    dataset_od_full = CIFAR10_NPY('image_data/Cifar10_1/cifar10.1_v4_data.npy', transform=transform)
    dataset_val1, dataset_val2, dataset_id_full = random_split(dataset_full, [n_val1, n_val2, len(dataset_full)-n_val1-n_val2])
    
    feats_val1 = []
    for X, _ in tqdm(dataset_val1):
        feat_val1 = backbone(X.to(device))
        feats_val1.append(feat_val1)
    feats_val1 = torch.stack(feats_val1)
    
    m_in_calib = []
    m_out_calib = []
    for X, _ in tqdm(dataset_val2):
        feat_val2 = backbone(X.to(device))
        m_in = cal_similarity(feat_val2, feat_val2)[mask].mean()       
        m_out = torch.vmap(lambda arg: cal_similarity(feat_val2, arg))(feats_val1).mean()
        m_in_calib.append(m_in)
        m_out_calib.append(m_out)
    m_in_calib = torch.tensor(m_in_calib)
    m_out_calib = torch.tensor(m_out_calib)
    gamma = torch.sqrt(m_in_calib.var() / m_out_calib.var())
    
    stats_id = []
    stats_od = []
    for i in tqdm(range(n_test)):
        index_id = np.random.randint(len(dataset_id_full))
        index_od = np.random.randint(len(dataset_od_full))
        X_id, _ = dataset_id_full[index_id]
        X_od, _ = dataset_od_full[index_od]
        feat_id = backbone(X_id.to(device))
        feat_od = backbone(X_od.to(device))
        m_in_id = cal_similarity(feat_id, feat_id)[mask].mean()       
        m_out_id = torch.vmap(lambda arg: cal_similarity(feat_id, arg))(feats_val1).mean()
        stat_id = m_in_id + gamma * m_out_id
        m_in_od = cal_similarity(feat_od, feat_od)[mask].mean()       
        m_out_od = torch.vmap(lambda arg: cal_similarity(feat_od, arg))(feats_val1).mean()
        stat_od = m_in_od + gamma * m_out_od
        stats_id.append(stat_id.item())
        stats_od.append(stat_od.item())
    draw_roc_curve(stats_id, stats_od)

def contrastive_two_sample_test_imagenet():
    view_transform = SimCLRViewTransform(input_size=224)
    class _transform:
        def __init__(self, num_tranforms: int):
            self.view_transform = view_transform
            self.n_transforms = num_tranforms
            
        def __call__(self, image):
            return torch.stack([self.view_transform(image) for _ in range(self.n_transforms)])
    
    cfg = OmegaConf.load('configs/config.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
    
    n_holdout = 5000
    n_trans = 50
    sample_size_test = 25
    n_test = 100
    mask = ~torch.eye(n_trans, dtype=bool)
    dataset_full = ImageFolder('image_data/imagenet-object-localization-challenge/ILSVRC\Data/CLS-LOC/val', transform=view_transform)
    dataset_od = ImageFolder('image_data/CW', transform=view_transform)
    dataset_holdout, dataset_id = random_split(dataset_full, [n_holdout, len(dataset_full)-n_holdout])
    
    # calculate holdout features
    feats_holdout = []
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=128)   
    for X, _ in tqdm(dataloader_holdout, desc='calculate hold out features'):
        feat = backbone(X.to(device))
        feats_holdout.append(feat)        
    feats_holdout = torch.cat(feats_holdout)   
      
    stats_od = []
    stats_id = []
    for i in tqdm(range(n_test), desc='two sample test'):
        dataset_od_test, _ = random_split(dataset_od, [sample_size_test, len(dataset_od)-sample_size_test])
        dataset_id_test, _ = random_split(dataset_id, [sample_size_test, len(dataset_id)-sample_size_test])
        
        # print('calculate aug_sim_diff')
        dataset_od_test.dataset.transform = _transform(num_tranforms=n_trans)
        dataset_id_test.dataset.dataset.transform = _transform(num_tranforms=n_trans)
        
        sims = []
        for X, _ in dataset_od_test:
            feat = backbone(X.to(device))
            sim = cal_similarity(feat, feat)[mask].mean()
            sims.append(sim.item())       
        aug_sims_od = torch.tensor(sims).mean()
        
        sims = []
        for X, _ in dataset_id_test:
            feat = backbone(X.to(device))
            sim = cal_similarity(feat, feat)[mask].mean()
            sims.append(sim.item())       
        aug_sims_id = torch.tensor(sims).mean()
        
        # print('calculate cross sample similarities')
        dataset_od_test.dataset.transform = view_transform
        dataset_id_test.dataset.dataset.transform = view_transform
        dataloader_od_test = DataLoader(dataset_od_test, batch_size=128)   
        dataloader_id_test = DataLoader(dataset_id_test, batch_size=128)
        
        feats_od = []
        for X, _ in dataloader_od_test:
            feat = backbone(X.to(device))
            feats_od.append(feat)        
        feats_od = torch.cat(feats_od)   
        cross_sims_od = cal_similarity(feats_od, feats_holdout).mean()
             
        feats_id = []
        for X, _ in dataloader_id_test:
            feat = backbone(X.to(device))
            feats_id.append(feat)        
        feats_id = torch.cat(feats_id)   
        cross_sims_id = cal_similarity(feats_id, feats_holdout).mean()
        
        stat_od = - aug_sims_od + cross_sims_od
        stat_id = - aug_sims_id + cross_sims_id
        stats_od.append(stat_od.item())
        stats_id.append(stat_id.item())
    print(f'rejection rate: {get_rejection_rate(stats_id, stats_od)}')
    draw_roc_curve(stats_id, stats_od)
    
def contrastive_two_sample_test_mnist(sample_size_test):
    view_transform = T.Compose([Convert('RGB'), SimCLRViewTransform(input_size=32, cj_prob=0)])
    cfg = OmegaConf.load('configs/config_mnist.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
        
    n_holdout = 5000
    n_trans = 50
    contrastive_transform = ContrastiveTransform(view_transform, n_trans=n_trans)
    n_test = 100
    mask = ~torch.eye(n_trans, dtype=bool)
    dataset_full = MNIST('image_data/MNIST', train=False, transform=view_transform)
    dataset_od = MNIST_FAKE('image_data/MNIST/Fake_MNIST_data_EP100_N10000.pckl', transform=view_transform)
    dataset_holdout, dataset_id = random_split(dataset_full, [n_holdout, len(dataset_full)-n_holdout])
    
    # calculate holdout features
    feats_holdout = []
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=128)   
    for X, _ in tqdm(dataloader_holdout, desc='calculate hold out features'):
        feat = backbone(X.to(device))
        feats_holdout.append(feat)        
    feats_holdout = torch.cat(feats_holdout)   
      
    stats_od = []
    stats_id = []
    for i in tqdm(range(n_test), desc='two sample test'):
        dataset_od_test, _ = random_split(dataset_od, [sample_size_test, len(dataset_od)-sample_size_test])
        dataset_id_test, _ = random_split(dataset_id, [sample_size_test, len(dataset_id)-sample_size_test])
        
        # print('calculate aug_sim_diff')
        dataset_od_test.dataset.transform = contrastive_transform
        dataset_id_test.dataset.dataset.transform = contrastive_transform
        
        sims = []
        for X, _ in dataset_od_test:
            feat = backbone(torch.stack(X).to(device))
            sim = cal_similarity(feat, feat)[mask].mean()
            sims.append(sim.item())       
        aug_sims_od = torch.tensor(sims).mean()
        
        sims = []
        for X, _ in dataset_id_test:
            feat = backbone(torch.stack(X).to(device))
            sim = cal_similarity(feat, feat)[mask].mean()
            sims.append(sim.item())       
        aug_sims_id = torch.tensor(sims).mean()
        
        # print('calculate cross sample similarities')
        dataset_od_test.dataset.transform = view_transform
        dataset_id_test.dataset.dataset.transform = view_transform
        dataloader_od_test = DataLoader(dataset_od_test, batch_size=128)   
        dataloader_id_test = DataLoader(dataset_id_test, batch_size=128)
        
        feats_od = []
        for X, _ in dataloader_od_test:
            feat = backbone(X.to(device))
            feats_od.append(feat)        
        feats_od = torch.cat(feats_od)   
        cross_sims_od = cal_similarity(feats_od, feats_holdout).mean()
             
        feats_id = []
        for X, _ in dataloader_id_test:
            feat = backbone(X.to(device))
            feats_id.append(feat)        
        feats_id = torch.cat(feats_id)   
        cross_sims_id = cal_similarity(feats_id, feats_holdout).mean()
        
        stat_od = - aug_sims_od + cross_sims_od
        stat_id = - aug_sims_id + cross_sims_id
        stats_od.append(stat_od.item())
        stats_id.append(stat_id.item())
    print(f'rejection rate: {get_rejection_rate(stats_id, stats_od)}')
    draw_roc_curve(stats_id, stats_od)

def mmd_two_sample_test_imagenet():
    view_transform = SimCLRViewTransform(input_size=224)
    cfg = OmegaConf.load('configs/config.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
    
    sample_size = 150
    n_test = 100
    
    dataset_full = ImageFolder('image_data/imagenet-object-localization-challenge/ILSVRC\Data/CLS-LOC/val', transform=view_transform)
    dataset_od_full = ImageFolder('image_data/CW', transform=view_transform)
    dataset_holdout, dataset_id_full = random_split(dataset_full, [sample_size, len(dataset_full)-sample_size])
    
   
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=sample_size)
    feats_holdout = []
    for X_holdout, _ in dataloader_holdout:
        feat_holdout = backbone(X_holdout.to(device))
        feats_holdout.append(feat_holdout)
    feats_holdout = torch.cat(feats_holdout)
    
    mmds_id = []
    mmds_od = []
    for i in tqdm(range(n_test), desc='two sample test'):
        dataset_od, _ = random_split(dataset_od_full, [sample_size, len(dataset_od_full)-sample_size])
        dataset_id, _ = random_split(dataset_id_full, [sample_size, len(dataset_id_full)-sample_size])
        dataloader_od = DataLoader(dataset_od, batch_size=sample_size)   
        dataloader_id = DataLoader(dataset_id, batch_size=sample_size)
        feats_od = []
        feats_id = []
        for (X_od, _), (X_id, _) in zip(dataloader_od, dataloader_id):
            feat_od = backbone(X_od.to(device))
            feat_id = backbone(X_id.to(device))
            feats_od.append(feat_od)
            feats_id.append(feat_id)
        feats_od = torch.cat(feats_od)
        feats_id = torch.cat(feats_id)
        
        _, _, mmd_id = mmd_permutation_test(feats_holdout, feats_id)
        _, _, mmd_od = mmd_permutation_test(feats_holdout, feats_od)  
        mmds_id.append(mmd_id)
        mmds_od.append(mmd_od)
    print(f'rejection rate: {get_rejection_rate(mmds_id, mmds_od)}')
    draw_roc_curve(mmds_id, mmds_od)

def mmd_two_sample_test_higgs():
    view_transform = T.Compose([
            ToTensor(),
            L2_Normalize()
        ])
    cfg = OmegaConf.load('configs/config_tab.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr_tab)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
    
    sample_size = 2000
    n_test = 100
    dataset_full = HIGGS('image_data/HIGGS/HIGGS_TST.pckl', class_id=0, transform=view_transform)
    dataset_od_full = HIGGS('image_data/HIGGS/HIGGS_TST.pckl', class_id=1, transform=view_transform)
    dataset_holdout, dataset_id_full = random_split(dataset_full, [sample_size, len(dataset_full)-sample_size])
    
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=sample_size)
    feats_holdout = []
    for X_holdout, _ in dataloader_holdout:
        feat_holdout = backbone(X_holdout.to(device))
        feats_holdout.append(feat_holdout)
    feats_holdout = torch.cat(feats_holdout)
    
    mmds_id = []
    mmds_od = []
    for i in tqdm(range(n_test), desc='two sample test'):
        dataset_od, _ = random_split(dataset_od_full, [sample_size, len(dataset_od_full)-sample_size])
        dataset_id, _ = random_split(dataset_id_full, [sample_size, len(dataset_id_full)-sample_size])
        dataloader_od = DataLoader(dataset_od, batch_size=sample_size)   
        dataloader_id = DataLoader(dataset_id, batch_size=sample_size)
        feats_od = []
        feats_id = []
        for (X_od, _), (X_id, _) in zip(dataloader_od, dataloader_id):
            feat_od = backbone(X_od.to(device))
            feat_id = backbone(X_id.to(device))
            feats_od.append(feat_od)
            feats_id.append(feat_id)
        feats_od = torch.cat(feats_od)
        feats_id = torch.cat(feats_id)
        
        _, _, mmd_id = mmd_permutation_test(feats_holdout, feats_id)
        _, _, mmd_od = mmd_permutation_test(feats_holdout, feats_od)  
        mmds_id.append(mmd_id)
        mmds_od.append(mmd_od)
    print(f'rejection rate: {get_rejection_rate(mmds_id, mmds_od)}')
    draw_roc_curve(mmds_id, mmds_od)

def contrastive_two_sample_test_higgs(sample_size_test):
    view_transform = T.Compose([
            ToTensor(),
            L2_Normalize()
        ])
    cfg = OmegaConf.load('configs/config_tab.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr_tab)
    # model = instantiate(cfg.classifier)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
        
    n_holdout = 5000
    n_trans = 50
    contrastive_transform = ContrastiveTransform(view_transform, n_trans=n_trans)
    n_test = 100
    mask = ~torch.eye(n_trans, dtype=bool)
    dataset_full = HIGGS('image_data/HIGGS/HIGGS_TST.pckl', class_id=0, transform=view_transform)
    dataset_od = HIGGS('image_data/HIGGS/HIGGS_TST.pckl', class_id=1, transform=view_transform)
    dataset_holdout, dataset_id = random_split(dataset_full, [n_holdout, len(dataset_full)-n_holdout])
    
    # calculate holdout features
    feats_holdout = []
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=256)   
    for X, _ in tqdm(dataloader_holdout, desc='calculate hold out features'):
        feat = backbone(X.to(device))
        feats_holdout.append(feat)        
    feats_holdout = torch.cat(feats_holdout)   
      
    stats_od = []
    stats_id = []
    for i in tqdm(range(n_test), desc='two sample test'):
        dataset_od_test, _ = random_split(dataset_od, [sample_size_test, len(dataset_od)-sample_size_test])
        dataset_id_test, _ = random_split(dataset_id, [sample_size_test, len(dataset_id)-sample_size_test])
        
        # print('calculate aug_sim_diff')
        dataset_od_test.dataset.transform = contrastive_transform
        dataset_id_test.dataset.dataset.transform = contrastive_transform
        
        sims = []
        for X, _ in dataset_od_test:
            feat = backbone(torch.stack(X).to(device))
            sim = cal_similarity(feat, feat)[mask].mean()
            sims.append(sim.item())       
        aug_sims_od = torch.tensor(sims).mean()
        
        sims = []
        for X, _ in dataset_id_test:
            feat = backbone(torch.stack(X).to(device))
            sim = cal_similarity(feat, feat)[mask].mean()
            sims.append(sim.item())       
        aug_sims_id = torch.tensor(sims).mean()
        
        # print('calculate cross sample similarities')
        dataset_od_test.dataset.transform = view_transform
        dataset_id_test.dataset.dataset.transform = view_transform
        dataloader_od_test = DataLoader(dataset_od_test, batch_size=128)   
        dataloader_id_test = DataLoader(dataset_id_test, batch_size=128)
        
        feats_od = []
        for X, _ in dataloader_od_test:
            feat = backbone(X.to(device))
            feats_od.append(feat)        
        feats_od = torch.cat(feats_od)   
        cross_sims_od = cal_similarity(feats_od, feats_holdout).mean()
             
        feats_id = []
        for X, _ in dataloader_id_test:
            feat = backbone(X.to(device))
            feats_id.append(feat)        
        feats_id = torch.cat(feats_id)   
        cross_sims_id = cal_similarity(feats_id, feats_holdout).mean()
        
        stat_od = - aug_sims_od + cross_sims_od
        stat_id = - aug_sims_id + cross_sims_id
        stats_od.append(stat_od.item())
        stats_id.append(stat_id.item())
    print(f'rejection rate: {get_rejection_rate(stats_id, stats_od)}')
    draw_roc_curve(stats_id, stats_od)


def contrastive_calibration(n_test):
    view_transform = SimCLRViewTransform(input_size=32)
    class _transform:
        def __init__(self, num_tranforms: int):
            self.view_transform = view_transform
            self.n_transforms = num_tranforms
            
        def __call__(self, image):
            return torch.stack([self.view_transform(image) for _ in range(self.n_transforms)])
    
    cfg = OmegaConf.load('configs/config_cifar.yml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.simclr)
    model._load_weights()
    model.to(device)
    deactivate_requires_grad(model)
    backbone = model.backbone 
    
    n_holdout = 5000
    n_trans = 50
    sample_size_calib = 500
    mask = ~torch.eye(n_trans, dtype=bool)
    dataset_full = CIFAR10('image_data/Cifar10', train=False, transform=view_transform)
    dataset_holdout, dataset_calib = random_split(dataset_full, [n_holdout, len(dataset_full)-n_holdout])
    
    # calculate holdout features
    feats_holdout = []
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=128)   
    for X, _ in tqdm(dataloader_holdout, desc='calculate hold out features'):
        feat = backbone(X.to(device))
        feats_holdout.append(feat)        
    feats_holdout = torch.cat(feats_holdout)   
    
    # calibrate 
    cross_sample_sims = []
    cross_aug_sims = []
    for _ in tqdm(range(n_test), desc='calibration test'):
        dataset_calib_sub, _ = random_split(dataset_calib, [sample_size_calib, len(dataset_calib) - sample_size_calib])
        
        dataset_calib_sub.dataset.dataset.transform = view_transform
        dataloader_calib = DataLoader(dataset_calib_sub, batch_size=128)
        feats_calib = []
        for X, _ in dataloader_calib:
            feat = backbone(X.to(device))
            feats_calib.append(feat)        
        feats_calib= torch.cat(feats_calib)   
        cross_sample_sim = cal_similarity(feats_calib, feats_holdout).mean()
        # print(f'in distribution cross-sim: {cross_sample_sim.item()}')
        cross_sample_sims.append(cross_sample_sim.item())
        
        sims = []
        dataset_calib_sub.dataset.dataset.transform = _transform(num_tranforms=n_trans)
        for X, _ in dataset_calib_sub:
            feat = backbone(X.to(device))
            sim = cal_similarity(feat, feat)[mask].mean()
            sims.append(sim.item())       
        cross_aug_sim = torch.tensor(sims).mean()
        # print(f'in distribution aug-sim: {cross_aug_sim.item()}')
        cross_aug_sims.append(cross_aug_sim.item())
    result = {
        'cross_sample_sims': np.array(cross_sample_sims),
        'cross_aug_sims': np.array(cross_aug_sims)
    }
    np.save(f'two_sample_test_results/contrastive_simclr_calib.npy', result)
    sns.kdeplot(cross_sample_sims, color='red', linewidth=2, label='cross_sample_sim')
    sns.kdeplot(cross_aug_sims, color='blue', linewidth=2, label='cross_aug_sim')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title("In Distribution Similarities")
    plt.legend(loc="upper right")
    plt.show()

def draw_results(method):
    assert method in ['mmd', 'contrastive']
    plt.figure(figsize=(10,6))
    sample_sizes = [50, 100, 200, 300, 500]
    # sample_sizes = [10, 30, 50, 100, 150]
    for sample_size in sample_sizes:
        data = np.load(f'two_sample_test_results/{method}_simclr_{sample_size}.npy', allow_pickle=True)   
        normal_scores = data.item()['stats_id']
        anomaly_scores = data.item()['stats_od']
        rejection_rate = get_rejection_rate(normal_scores, anomaly_scores)
        y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
        y_score = np.concatenate([normal_scores, anomaly_scores])      
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{sample_size} samples: auroc={roc_auc:.2f}. rejection_rate(alpha=0.05)={rejection_rate:.2f}')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve of {method} test on CIFAR10/CIFAR10.1')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    # calculate_features()
    # similarity_test()
    # similarity_aug()
    # similarity_cross()
    # contrastive_two_sample_test(50)
    # mmd_two_sample_test(500)
    # cadet_test()
    # contrastive_two_sample_test_imagenet()
    # mmd_two_sample_test_imagenet()
    # mmd_two_sample_test_mnist()
    contrastive_two_sample_test_mnist(4)
    # contrastive_calibration(100)

    # mmd_two_sample_test_higgs()
    # contrastive_two_sample_test_higgs(1000)
    
    # draw_results('mmd')
    