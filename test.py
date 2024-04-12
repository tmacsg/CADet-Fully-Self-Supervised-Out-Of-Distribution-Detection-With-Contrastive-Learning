from omegaconf import OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
from lightly.models.utils import deactivate_requires_grad
from utils.stat_utils import draw_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test_classifier_imagenet():
    cfg = OmegaConf.load('configs/config.yml')
    cfg.imagenet_data_module.args.mode = 'supervised'
    cfg.classifier.args.mode = 'validate'
    model = instantiate(cfg.classifier)
    data_module = instantiate(cfg.imagenet_data_module)
    trainer = pl.Trainer()
    trainer.validate(model, data_module)

def test_simclr_imagenet():
    cfg = OmegaConf.load('configs/config.yml')
    cfg.imagenet_data_module.args.mode = 'supervised'
    cfg.simclr.args.mode = 'validate'
    model = instantiate(cfg.simclr)
    data_module = instantiate(cfg.imagenet_data_module)
    trainer = pl.Trainer()
    trainer.validate(model, data_module)

def test_classifier_cifar():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    cfg.cifar_data_module.args.mode = 'supervised'
    cfg.classifier.args.mode = 'validate'
    model = instantiate(cfg.classifier)
    data_module = instantiate(cfg.cifar_data_module)
    trainer = pl.Trainer()
    trainer.validate(model, data_module)
    
def test_classifier_mnist():
    cfg = OmegaConf.load('configs/config_mnist.yml')
    cfg.mnist_data_module.args.mode = 'supervised'
    cfg.classifier.args.mode = 'validate'
    model = instantiate(cfg.classifier)
    data_module = instantiate(cfg.mnist_data_module)
    trainer = pl.Trainer()
    trainer.validate(model, data_module)
    
def eval_simclr_imagenet():
    cfg = OmegaConf.load('configs/config.yml')
    cfg.imagenet_data_module.args.mode = 'supervised'
    cfg.simclr.args.mode = 'linear_eval'
    cfg.simclr.args.lr = 0.3
    cfg.simclr.args.max_epochs = 40
    n_devices = 1
    accelerator = 'gpu'
    accumulate_grad_batches = 1
    strategy = DDPStrategy(find_unused_parameters=True, process_group_backend='gloo')
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_eval_imagenet' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="cifar_ckpts", save_top_k=1, save_last=False,
                                          monitor="val_acc_cls", mode='max')
    model = instantiate(cfg.simclr)
    data_module = instantiate(cfg.imagenet_data_module)
    trainer = pl.Trainer(devices=n_devices,
                         accelerator=accelerator,
                         accumulate_grad_batches=accumulate_grad_batches,
                         strategy=strategy,
                         max_epochs=cfg.simclr.args.max_epochs,
                         logger=logger,
                         callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)

def eval_simclr_cifar():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    cfg.cifar_data_module.args.mode = 'supervised'
    cfg.simclr.args.mode = 'linear_eval'
    cfg.simclr.args.lr = 0.3
    cfg.simclr.args.max_epochs = 30
    n_devices = 1
    accelerator = 'gpu'
    accumulate_grad_batches = 1
    strategy = DDPStrategy(find_unused_parameters=True, process_group_backend='gloo')
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_eval_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="cifar_ckpts", save_top_k=1, save_last=False,
                                          monitor="val_acc_cls", mode='max')
    model = instantiate(cfg.simclr)
    data_module = instantiate(cfg.cifar_data_module)
    trainer = pl.Trainer(devices=n_devices,
                         accelerator=accelerator,
                         accumulate_grad_batches=accumulate_grad_batches,
                         strategy=strategy,
                         max_epochs=cfg.simclr.args.max_epochs,
                         logger=logger,
                         callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)
    
def eval_simclr_mnist():
    cfg = OmegaConf.load('configs/config_mnist.yml')
    cfg.mnist_data_module.args.mode = 'supervised'
    cfg.simclr.args.mode = 'linear_eval'
    cfg.simclr.args.lr = 0.3
    cfg.simclr.args.max_epochs = 30
    n_devices = 1
    accelerator = 'gpu'
    accumulate_grad_batches = 1
    strategy = DDPStrategy(find_unused_parameters=True, process_group_backend='gloo')
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_eval_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="mnist_ckpts", save_top_k=1, save_last=False,
                                          monitor="val_acc_cls", mode='max')
    model = instantiate(cfg.simclr)
    data_module = instantiate(cfg.mnist_data_module)
    trainer = pl.Trainer(devices=n_devices,
                         accelerator=accelerator,
                         accumulate_grad_batches=accumulate_grad_batches,
                         strategy=strategy,
                         max_epochs=cfg.simclr.args.max_epochs,
                         logger=logger,
                         callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)

def test_simclr_cifar():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    cfg.cifar_data_module.args.mode = 'supervised'
    cfg.simclr.args.mode = 'validate'
    model = instantiate(cfg.simclr)
    data_module = instantiate(cfg.cifar_data_module)
    trainer = pl.Trainer()
    trainer.validate(model, data_module)

def test_mmd_imagenet():
    cfg = OmegaConf.load('configs/config.yml')
    cfg.imagenet_data_module.args.mode = 'mmd'
    cfg.mmd.args.kernel = 'guassian'
    cfg.mmd.args.clean_calib = True
    cfg.mmd.args.image_set_q = 'cw'
    model = instantiate(cfg.mmd)
    data_module = instantiate(cfg.imagenet_data_module)
    trainer = pl.Trainer(devices=1)
    trainer.test(model, data_module)

def test_mmd_cifar():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    cfg.cifar_data_module.args.mode = 'mmd'
    cfg.mmd.args.kernel = 'cosine'
    cfg.mmd.args.clean_calib = False
    cfg.mmd.args.image_set_q = 'fgsm'
    model = instantiate(cfg.mmd)
    data_module = instantiate(cfg.cifar_data_module)
    trainer = pl.Trainer(devices=1)
    trainer.test(model, data_module)

def test_mmd_mnist():
    cfg = OmegaConf.load('configs/config_mnist.yml')
    cfg.mnist_data_module.args.mode = 'mmd'
    cfg.mmd.args.kernel = 'cosine'
    cfg.mmd.args.clean_calib = True
    cfg.mmd.args.image_set_q = 'mnist_fake'
    model = instantiate(cfg.mmd)
    data_module = instantiate(cfg.mnist_data_module)
    trainer = pl.Trainer(devices=1)
    trainer.test(model, data_module)


def test_cadet_imagenet():
    cfg = OmegaConf.load('configs/config.yml')
    cfg.imagenet_data_module.args.mode = 'cadet'
    model = instantiate(cfg.cadet)
    data_module = instantiate(cfg.imagenet_data_module)
    trainer = pl.Trainer()
    trainer.test(model, data_module)

def test_cadet_cifar():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    cfg.cifar_data_module.args.mode = 'cadet'
    model = instantiate(cfg.cadet)
    data_module = instantiate(cfg.cifar_data_module)
    trainer = pl.Trainer()
    trainer.test(model, data_module)

def test_mmd_ss_cifar():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    cfg.cifar_data_module.args.mode = 'mmd_ss'
    cfg.mmd.args.kernel = 'cosine'
    cfg.mmd.args.clean_calib = True
    cfg.mmd.args.image_set_q = 'cifar10_1'
    model = instantiate(cfg.mmd_ss)
    data_module = instantiate(cfg.cifar_data_module)
    trainer = pl.Trainer(devices=1)
    trainer.test(model, data_module)
    
def test_cadet_ss_cifar():
    from utils.data_utils import CIFAR10_NPY, CIFAR10, CADetTransform_CIFAR
    from torch.utils.data import DataLoader, random_split, Subset
    import torch.nn.functional as F
    from tqdm import tqdm
    import numpy as np
    from sklearn import metrics
    import torchvision.transforms as T
    from lightly.transforms.utils import IMAGENET_NORMALIZE
    
    def _cal_similarity(input0, input1):
        # return torch.exp(-torch.cdist(input0, input1, p=2))
        out0 = F.normalize(input0, dim=1)
        out1 = F.normalize(input1, dim=1)
        return out0 @ out1.t()
    class _transform:
        def __init__(self, num_tranforms: int):
            self.view_transform = T.Compose([
                T.Resize([32,32], interpolation=T.InterpolationMode.BILINEAR),
                T.RandomResizedCrop(size=32, scale=(0.75, 0.75)),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
            ])
            self.n_transforms = num_tranforms

        def __call__(self, image):
            return torch.stack([self.view_transform(image) for _ in range(self.n_transforms)])
    
    n_trans = 50
    n_val1 = 300
    n_val2 = 2000
    n_tests = 100
    transform=_transform(num_tranforms=n_trans)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load('configs/config_cifar.yml')
    model = instantiate(cfg.simclr)
    model._load_weights()
    deactivate_requires_grad(model)
    backbone = model.backbone
    backbone.to(device)
    
    val_dataset = CIFAR10('image_data/Cifar10', train=False, transform=transform)
    val_split = [n_val1, n_val2, len(val_dataset) - n_val1 - n_val2]
    dataset_1, dataset_2, in_dist_dataset = random_split(val_dataset, val_split)
    test_dataset = CIFAR10_NPY('image_data/Cifar10_1/cifar10.1_v4_data.npy', transform=transform) 
    # test_dataset = CIFAR10_NPY('image_data/Cifar10/cifar10_val_CW.npy', transform=transform) 

    # calculate X_feat_1
    X_feats = []
    dataloader_1 = DataLoader(dataset_1, batch_size=1, shuffle=False)
    for batch_idx, (X, _) in enumerate(tqdm(dataloader_1, desc="calculate x_feat")):
        X = X[0].to(device)
        X_feat = backbone(X)
        X_feats.append(X_feat)
    X_feats = torch.stack(X_feats)
        
    # calibrate m_in, m_out, gamma
    m_ins = []
    m_outs = []
    dataloader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False)
    for batch_idx, (X, _) in enumerate(tqdm(dataloader_2, desc="calibrate m and gamma")):    
        X = X[0].to(device)
        X_feat = backbone(X)
        
        intra_sim = _cal_similarity(X_feat, X_feat)
        m_in = (intra_sim.sum() - intra_sim.trace()) / (n_trans * (n_trans - 1))
        outer_sim = torch.vmap(lambda arg: _cal_similarity(arg, X_feat))(X_feats)
        m_out = outer_sim.sum() / (n_val1 * n_trans * n_trans)      
        m_ins.append(m_in)
        m_outs.append(m_out)
        
    m_ins = torch.tensor(m_ins, device=device)
    m_outs = torch.tensor(m_outs, device=device)
    gamma = (m_ins.var() / m_outs.var()).sqrt()    
    scores = m_ins + gamma * m_outs
    
    print(f'in distribution m_in: mean {m_ins.mean().item()} var {m_ins.var().item()}')
    print(f'in distribution m_out: mean {m_outs.mean().item()} var {m_outs.var().item()}')
    print('gamma ', gamma.item())
    
        
    p_id = []
    p_od = []
    for i in tqdm(range(n_tests), desc="test"):
        id_data = in_dist_dataset[np.random.randint(0,len(in_dist_dataset))][0]
        od_data = test_dataset[np.random.randint(0,len(test_dataset))][0]
        X_id_feat = backbone(id_data.to(device))       
        intra_sim_id = _cal_similarity(X_id_feat, X_id_feat)
        m_in_id = (intra_sim_id.sum() - intra_sim_id.trace()) / (n_trans * (n_trans - 1))        
        outer_sim_id = torch.vmap(lambda arg: _cal_similarity(arg, X_id_feat))(X_feats)
        m_out_id = outer_sim_id.sum() / (n_val1 * n_trans * n_trans)    
        score_id = m_in_id + gamma * m_out_id
        p_id.append((torch.sum(score_id > scores).item() + 1) / (len(scores) + 1))  
        
        X_od_feat = backbone(od_data.to(device))
        intra_sim_od = _cal_similarity(X_od_feat, X_od_feat)
        m_in_od = (intra_sim_od.sum() - intra_sim_od.trace()) / (n_trans * (n_trans - 1))        
        outer_sim_od = torch.vmap(lambda arg: _cal_similarity(arg, X_od_feat))(X_feats)
        m_out_od = outer_sim_od.sum() / (n_val1 * n_trans * n_trans)    
        score_od = m_in_od + gamma * m_out_od
        p_od.append((torch.sum(score_od > scores).item() + 1) / (len(scores) + 1)) 
        

    labels = np.zeros(n_tests * 2)
    labels[n_tests:] = 1
    p_id.extend(p_od)
    fpr, tpr, _ = metrics.roc_curve(labels, [1 - p for p in p_id])
    auroc = metrics.auc(fpr, tpr)
    print(auroc)

def test_cadet_ss_imagenet():
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, random_split, Subset, RandomSampler
    import torch.nn.functional as F
    from tqdm import tqdm
    import numpy as np
    from sklearn import metrics
    import torchvision.transforms as T
    from lightly.transforms import SimCLRTransform, SimCLRViewTransform
    from lightly.transforms.utils import IMAGENET_NORMALIZE
    from pytorch_lightning.utilities.combined_loader import CombinedLoader
    import matplotlib.pyplot as plt
    
    def _cal_similarity(input0, input1):
        # return torch.exp(-torch.cdist(input0, input1, p=2))
        out0 = F.normalize(input0, dim=1)
        out1 = F.normalize(input1, dim=1)
        return out0 @ out1.t()
        
    class Convert:
        def __init__(self, mode='RGB'):
            self.mode = mode

        def __call__(self, image):
            return image.convert(self.mode)
    class _transform:
        def __init__(self, num_tranforms: int):
            self.view_transform = T.Compose([
                Convert('RGB'),
                T.Resize([256,256], interpolation=T.InterpolationMode.BILINEAR),
                T.RandomResizedCrop(size=224, scale=(0.75, 0.75)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std'])
            ])
            self.n_transforms = num_tranforms

        def __call__(self, image):
            return torch.stack([self.view_transform(image) for _ in range(self.n_transforms)])
        
    # n_trans = 5
    # n_val1 = 20
    # n_val2 = 20
    # n_tests = 10
    n_trans = 50
    n_val1 = 300
    n_val2 = 2000
    n_tests = 100
    transform = _transform(num_tranforms=n_trans)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load('configs/config.yml')
    model = instantiate(cfg.simclr)
    model._load_weights()
    deactivate_requires_grad(model)
    backbone = model.backbone
    backbone.to(device)
    
    val_dataset = ImageFolder('image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val', 
                              transform=transform)
    val_split = [n_val1, n_val2, len(val_dataset) - n_val1 - n_val2]
    dataset_1, dataset_2, in_dist_dataset = random_split(val_dataset, val_split)
    # test_dataset = ImageFolder('image_data/imagenet-o', transform=transform) 
    ood_dataset = ImageFolder('image_data/pgd', transform=transform) 

    # calculate X_feat_1
    X_1_feats = []
    dataloader_1 = DataLoader(dataset_1, batch_size=1, shuffle=False)
    for batch_idx, (X, _) in enumerate(tqdm(dataloader_1, desc="calculate x_feat")):
        X = X[0].to(device)
        
        # _, axes = plt.subplots(1, len(X), figsize=(len(X), 2))
        # for i in range(len(X)):
        #     axes[i].imshow(X[i].cpu().permute(1, 2, 0))
        #     axes[i].axis('off')
        # plt.tight_layout()
        # plt.show()
        
        X_1_feat = backbone(X)
        X_1_feats.append(X_1_feat)
    X_1_feats = torch.stack(X_1_feats)
        
    # calibrate m_in, m_out, gamma
    m_ins = []
    m_outs = []
    dataloader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False)
    for batch_idx, (X, _) in enumerate(tqdm(dataloader_2, desc="calibrate m and gamma")):    
        X = X[0].to(device)
        X_2_feat = backbone(X)
                
        # _, axes = plt.subplots(1, len(X), figsize=(len(X), 2))
        # for i in range(len(X)):
        #     axes[i].imshow(X[i].cpu().permute(1, 2, 0))
        #     axes[i].axis('off')
        # plt.tight_layout()
        # plt.show()
        
        intra_sim = _cal_similarity(X_2_feat, X_2_feat)
        m_in = (intra_sim.sum() - intra_sim.trace()) / (n_trans * (n_trans - 1))
        outer_sim = torch.vmap(lambda arg: _cal_similarity(X_2_feat, arg))(X_1_feats)
        m_out = outer_sim.sum() / (n_val1 * n_trans * n_trans)      
        m_ins.append(m_in)
        m_outs.append(m_out)    
        
    m_ins = torch.tensor(m_ins, device=device)
    m_outs = torch.tensor(m_outs, device=device)
    gamma = (m_ins.var() / m_outs.var()).sqrt()    
    scores = m_ins + gamma * m_outs
        
    p_id = []
    p_od = []
    m_ins_od = []
    m_outs_od = []
    
    sampler_in = RandomSampler(in_dist_dataset, replacement=False, num_samples=n_tests)
    sampler_out = RandomSampler(ood_dataset, replacement=False, num_samples=n_tests)
    dataloder_in = DataLoader(in_dist_dataset, sampler=sampler_in)
    dataloder_out = DataLoader(ood_dataset, sampler=sampler_out)
    
    scores_id = []
    scores_od = []
    for _, ((id_data, _), (od_data, _)) in enumerate(tqdm(zip(dataloder_in, dataloder_out), 'test')):
        id_data = id_data[0].to(device)
        od_data = od_data[0].to(device)

        X_id_feat = backbone(id_data)       
        intra_sim_id = _cal_similarity(X_id_feat, X_id_feat)
        m_in_id = (intra_sim_id.sum() - intra_sim_id.trace()) / (n_trans * (n_trans - 1))        
        outer_sim_id = torch.vmap(lambda arg: _cal_similarity(X_id_feat, arg))(X_1_feats)
        m_out_id = outer_sim_id.sum() / (n_val1 * n_trans * n_trans)    
        score_id = m_in_id + gamma * m_out_id
        scores_id.append(score_id.item())
        p_id.append((torch.sum(score_id > scores).item() + 1) / (len(scores) + 1))  
        
        X_od_feat = backbone(od_data)
        intra_sim_od = _cal_similarity(X_od_feat, X_od_feat)
        m_in_od = (intra_sim_od.sum() - intra_sim_od.trace()) / (n_trans * (n_trans - 1))        
        outer_sim_od = torch.vmap(lambda arg: _cal_similarity(X_od_feat, arg))(X_1_feats)
        m_out_od = outer_sim_od.sum() / (n_val1 * n_trans * n_trans)    
        score_od = m_in_od + gamma * m_out_od
        scores_od.append(score_od.item())
        p_od.append((torch.sum(score_od > scores).item() + 1) / (len(scores) + 1)) 
        
        m_ins_od.append(m_in_od)
        m_outs_od.append(m_out_od)
        
    draw_roc_curve(scores_id, scores_od)

    # m_ins_od = torch.tensor(m_ins_od, device=device)
    # m_outs_od = torch.tensor(m_outs_od, device=device)

    # labels = np.zeros(n_tests * 2)
    # labels[n_tests:] = 1
    # p_id.extend(p_od)
    # fpr, tpr, _ = metrics.roc_curve(labels, [1 - p for p in p_id])
    # auroc = metrics.auc(fpr, tpr)
    
    # print(f'in distribution m_in: mean {m_ins.mean().item()} var {m_ins.var().item()}')
    # print(f'in distribution m_out: mean {m_outs.mean().item()} var {m_outs.var().item()}')
    # print('gamma ', gamma.item())
    # print(f'out distribution m_in: mean {m_ins_od.mean().item()} var {m_ins_od.var().item()}')
    # print(f'out distribution m_out: mean {m_outs_od.mean().item()} var {m_outs_od.var().item()}')
    # print(f'auroc: {auroc}')

def eval_simclr_breast_cancer():
    cfg = OmegaConf.load('configs/config_tabular.yml')
    cfg.simclr_tab.args.mode = 'linear_eval'
    cfg.simclr_tab.args.input_dim = 30
    cfg.simclr_tab.args.feature_dim = 64
    cfg.simclr_tab.args.output_dim = 32
    cfg.simclr_tab.args.lr = 0.3
    cfg.simclr_tab.args.max_epochs = 100
    model = instantiate(cfg.simclr_tab)
    cfg.breast_cancer_data_module.args.mode = 'supervised'
    data_module = instantiate(cfg.breast_cancer_data_module)
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_eval_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="breast_cancer_ckpts", save_top_k=1, save_last=True,
                                          monitor="val_acc_cls", mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    n_devices = 1
    accelerator = 'gpu'
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    trainer = pl.Trainer(devices=n_devices, accelerator=accelerator,
                          max_epochs = cfg.simclr_tab.args.max_epochs,
                          logger = logger,
                          callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)

def eval_simclr_income():
    cfg = OmegaConf.load('configs/config_tabular.yml')
    cfg.simclr_tab.args.mode = 'linear_eval'
    cfg.simclr_tab.args.input_dim = 104
    cfg.simclr_tab.args.feature_dim = 256
    cfg.simclr_tab.args.output_dim = 128
    cfg.simclr_tab.args.lr = 0.01
    cfg.simclr_tab.args.max_epochs = 100
    model = instantiate(cfg.simclr_tab)
    cfg.income_data_module.args.mode = 'supervised'
    cfg.income_data_module.args.batch_size = 128
    data_module = instantiate(cfg.income_data_module)
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_eval_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="income_ckpts", save_top_k=1, save_last=True,
                                          monitor="val_acc_cls", mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    n_devices = 1
    accelerator = 'gpu'
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    trainer = pl.Trainer(devices=n_devices, accelerator=accelerator,
                          max_epochs = cfg.simclr_tab.args.max_epochs,
                          logger = logger,
                          callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)
    
def eval_simclr_gesture_phase():
    cfg = OmegaConf.load('configs/config_tabular.yml')
    cfg.simclr_tab.args.mode = 'linear_eval'
    cfg.simclr_tab.args.num_classes = 5
    cfg.simclr_tab.args.input_dim = 50
    cfg.simclr_tab.args.feature_dim = 256
    cfg.simclr_tab.args.output_dim = 128
    cfg.simclr_tab.args.lr = 0.01
    cfg.simclr_tab.args.max_epochs = 200
    model = instantiate(cfg.simclr_tab)
    cfg.gesture_phase_data_module.args.mode = 'supervised'
    data_module = instantiate(cfg.gesture_phase_data_module)
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_eval_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="gesture_phase_ckpts", save_top_k=1, save_last=True,
                                          monitor="val_acc_cls", mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    n_devices = 1
    accelerator = 'gpu'
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    trainer = pl.Trainer(devices=n_devices, accelerator=accelerator,
                          max_epochs = cfg.simclr_tab.args.max_epochs,
                          logger = logger,
                          callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)
    
def eval_simclr_robot_wall():
    cfg = OmegaConf.load('configs/config_tabular.yml')
    cfg.simclr_tab.args.mode = 'linear_eval'
    cfg.simclr_tab.args.num_classes = 4
    cfg.simclr_tab.args.input_dim = 24
    cfg.simclr_tab.args.feature_dim = 256
    cfg.simclr_tab.args.output_dim = 128
    cfg.simclr_tab.args.lr = 0.01
    cfg.simclr_tab.args.max_epochs = 200
    model = instantiate(cfg.simclr_tab)
    cfg.robot_wall_data_module.args.mode = 'supervised'
    cfg.robot_wall_data_module.args.batch_size = 128
    data_module = instantiate(cfg.robot_wall_data_module)
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_eval_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="robot_wall_ckpts", save_top_k=1, save_last=True,
                                          monitor="val_acc_cls", mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    n_devices = 1
    accelerator = 'gpu'
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    trainer = pl.Trainer(devices=n_devices, accelerator=accelerator,
                          max_epochs = cfg.simclr_tab.args.max_epochs,
                          logger = logger,
                          callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)
    
    

def eval_simclr_theorem():
    cfg = OmegaConf.load('configs/config_tabular.yml')
    cfg.simclr_tab.args.mode = 'linear_eval'
    cfg.simclr_tab.args.num_classes = 6
    cfg.simclr_tab.args.input_dim = 51
    cfg.simclr_tab.args.feature_dim = 256
    cfg.simclr_tab.args.output_dim = 128
    cfg.simclr_tab.args.lr = 0.001
    cfg.simclr_tab.args.max_epochs = 200
    model = instantiate(cfg.simclr_tab)
    cfg.theorem_data_module.args.mode = 'supervised'
    cfg.theorem_data_module.args.batch_size = 64
    data_module = instantiate(cfg.theorem_data_module)
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_eval_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="theorem_ckpts", save_top_k=1, save_last=True,
                                          monitor="val_acc_cls", mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    n_devices = 1
    accelerator = 'gpu'
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    trainer = pl.Trainer(devices=n_devices, accelerator=accelerator,
                          max_epochs = cfg.simclr_tab.args.max_epochs,
                          logger = logger,
                          callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)

def eval_simclr_tabular():
    dataset_name = 'adult_income'
    assert dataset_name in ['breast_cancer', 'adult_income', 'gesture_phase', 'robot_wall', 
                            'theorem', 'obesity', 'ozone', 'texture', 'dna']
    cfg = OmegaConf.load('configs/config_tabular.yml')
    dataset_configs = {
        'breast_cancer': cfg.breast_cancer,
        'adult_income': cfg.adult_income,
        'gesture_phase': cfg.gesture_phase,
        'robot_wall': cfg.robot_wall,
        'theorem': cfg.theorem,
        'obesity': cfg.obesity,
        'ozone': cfg.ozone,
        'texture': cfg.texture,
        'dna': cfg.dna
    }
    dataset_config = dataset_configs[dataset_name]
    dataset_config.data_module.args.mode = 'supervised'
    cfg.simclr_tab.args.mode = 'linear_eval'
    cfg.simclr_tab.args.num_classes = dataset_config.num_classes
    cfg.simclr_tab.args.input_dim = dataset_config.input_dim
    cfg.simclr_tab.args.feature_dim = dataset_config.feature_dim
    cfg.simclr_tab.args.output_dim = dataset_config.output_dim
    cfg.simclr_tab.args.ckpt_path = dataset_config.ckpt_path
    cfg.simclr_tab.args.max_epochs = 200
    cfg.simclr_tab.args.lr = 0.3
    
    model = instantiate(cfg.simclr_tab)
    data_module = instantiate(dataset_config.data_module)
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_eval_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath=f"{dataset_name}_ckpts", save_top_k=1, save_last=True,
                                          monitor="val_acc_cls", mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    n_devices = 1
    accelerator = 'gpu'
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    trainer = pl.Trainer(devices=n_devices, accelerator=accelerator,
                          max_epochs = cfg.simclr_tab.args.max_epochs,
                          logger = logger,
                          callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)

def logistic_regression():
    dataset_name = 'robot_wall'
    assert dataset_name in ['breast_cancer', 'adult_income', 'gesture_phase', 'robot_wall', 
                            'theorem', 'obesity', 'ozone', 'texture', 'dna']
    cfg = OmegaConf.load('configs/config_tabular.yml')
    dataset_configs = {
        'breast_cancer': cfg.breast_cancer,
        'adult_income': cfg.adult_income,
        'gesture_phase': cfg.gesture_phase,
        'robot_wall': cfg.robot_wall,
        'theorem': cfg.theorem,
        'obesity': cfg.obesity,
        'ozone': cfg.ozone,
        'texture': cfg.texture,
        'dna': cfg.dna
    }
    dataset_config = dataset_configs[dataset_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.simclr_tab.args.num_classes = dataset_config.num_classes
    cfg.simclr_tab.args.input_dim = dataset_config.input_dim
    cfg.simclr_tab.args.feature_dim = dataset_config.feature_dim
    cfg.simclr_tab.args.output_dim = dataset_config.output_dim
    cfg.simclr_tab.args.ckpt_path = dataset_config.ckpt_path
    model = instantiate(cfg.simclr_tab)
    model._load_weights()
    deactivate_requires_grad(model)
    backbone = model.backbone
    backbone.to(device)
    dataset_train = instantiate(dataset_config.train)
    dataset_test = instantiate(dataset_config.test)
    train_data, train_target = dataset_train[:]
    test_data, test_target = dataset_test[:]
    X_train = torch.tensor(train_data).to(device)
    X_test = torch.tensor(test_data).to(device)
    train_embeddings = backbone(X_train).cpu()
    test_embeddings = backbone(X_test).cpu()
    
    ### raw feature regression
    clf = LogisticRegression()
    clf.fit(train_data, train_target)
    predictions = clf.predict(test_data)
    print(classification_report(test_target, predictions))
    cm = confusion_matrix(test_target, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig1, ax = plt.subplots(figsize=(5, 5))
    fig1.suptitle("Raw Features Logistic Regression")
    disp.plot(ax=ax)
    
    ### embeddings regression
    clf.fit(train_embeddings, train_target)
    predictions = clf.predict(test_embeddings)
    print(classification_report(test_target, predictions))
    cm = confusion_matrix(test_target, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig2, ax = plt.subplots(figsize=(5, 5))
    fig2.suptitle("Embeddings Logistic Regression")
    disp.plot(ax=ax)
    
    ### tSNE for raw features
    # tsne1 = TSNE(n_components=2)
    # reduced1 = tsne1.fit_transform(train_data)
    # positive = train_target == 1
    # fig3, ax3 = plt.subplots(figsize=(5, 5))
    # fig3.suptitle("Raw Feature tSNE")
    # ax3.scatter(reduced1[positive, 0], reduced1[positive, 1], label="positive")
    # ax3.scatter(reduced1[~positive, 0], reduced1[~positive, 1], label="negative")
    
    ### tSNE for embedding
    # tsne = TSNE(n_components=2)
    # reduced = tsne.fit_transform(train_embeddings)
    # positive = train_target == 1
    # fig4, ax4 = plt.subplots(figsize=(5, 5))
    # fig4.suptitle("Embeddings tSNE")
    # ax4.scatter(reduced[positive, 0], reduced[positive, 1], label="positive")
    # ax4.scatter(reduced[~positive, 0], reduced[~positive, 1], label="negative")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    pl.seed_everything(42)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    # test_classifier_cifar()
    # test_simclr_cifar()
    # test_mmd_cifar()
    # test_cadet_cifar()
    
    # test_classifier_mnist()
    # eval_simclr_mnist()
    # test_mmd_mnist()
    
    # test_classifier_imagenet()
    # test_simclr_imagenet()
    # test_mmd_imagenet()
    # test_cadet_imagenet()
    
    # test_cadet_ss_cifar()
    # test_cadet_ss_imagenet()
    # eval_simclr_breast_cancer()
    # eval_simclr_income()
    # eval_simclr_gesture_phase()
    # eval_simclr_robot_wall()
    # eval_simclr_theorem()
    # eval_simclr_tabular()
    logistic_regression()