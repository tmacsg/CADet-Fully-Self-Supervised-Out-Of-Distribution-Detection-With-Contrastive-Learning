from omegaconf import OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch

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
    cfg.mmd.args.image_set_q = 'pgd'
    model = instantiate(cfg.mmd)
    data_module = instantiate(cfg.imagenet_data_module)
    trainer = pl.Trainer(devices=1)
    trainer.test(model, data_module)

def test_mmd_cifar():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    cfg.cifar_data_module.args.mode = 'mmd'
    cfg.mmd.args.kernel = 'guassian'
    cfg.mmd.args.clean_calib = True
    cfg.mmd.args.image_set_q = 'cifar10_1'
    model = instantiate(cfg.mmd)
    data_module = instantiate(cfg.cifar_data_module)
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

if __name__ == '__main__':
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    pl.seed_everything(42)
    # test_classifier_cifar()
    # test_simclr_cifar()
    # test_cadet_cifar()
    # test_mmd_cifar()
    # test_mmd_ss_cifar()
    
    # test_classifier_imagenet()
    # test_simclr_imagenet()
    # test_mmd_imagenet()
    test_cadet_imagenet()
    
    
   
    
    
    