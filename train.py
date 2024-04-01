from omegaconf import OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import os

def train_simclr_cifar():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    cfg.cifar_data_module.args.mode = 'unsupervised'
    cfg.simclr.args.lr = 1.2
    cfg.simclr.args.max_epochs = 800
    model = instantiate(cfg.simclr)
    data_module = instantiate(cfg.cifar_data_module)
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_' +'{epoch:02d}_{train_loss_ssl:.4f}',
                                          dirpath="cifar_ckpts", save_top_k=1, save_last=True,
                                          monitor="train_loss_ssl", mode='min')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    n_devices = 2
    accelerator = 'gpu'
    accumulate_grad_batches = 2
    strategy = DDPStrategy(find_unused_parameters=True, process_group_backend='gloo')
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    trainer = pl.Trainer(devices=n_devices, accelerator=accelerator, strategy=strategy,
                          accumulate_grad_batches=accumulate_grad_batches,
                          max_epochs = cfg.simclr.args.max_epochs,
                          logger = logger,
                          callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)

def train_cls_cifar():
    cfg = OmegaConf.load('configs/config_cifar.yml')
    cfg.cifar_data_module.args.mode = 'supervised'
    cfg.classifier.args.lr = 0.3
    cfg.classifier.args.max_epochs = 200
    model = instantiate(cfg.classifier)
    data_module = instantiate(cfg.cifar_data_module)

    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_cls_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="cifar_ckpts", save_top_k=1, save_last=True,
                                          monitor="val_acc_cls", mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    # n_devices = 1
    # accelerator = 'gpu'
    # accumulate_grad_batches = 1
    # strategy = DDPStrategy(find_unused_parameters=True, process_group_backend='gloo')
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    # trainer = pl.Trainer(devices=n_devices, accelerator=accelerator, strategy=strategy,
    #                       accumulate_grad_batches=accumulate_grad_batches,
    #                       max_epochs = cfg.classifier.args.max_epochs,
    #                       logger = logger,
    #                       callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer = pl.Trainer(max_epochs = cfg.classifier.args.max_epochs, 
                         accelerator = 'gpu',
                         logger = logger,
                         callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)

def train_simclr_imagenet():
    cfg = OmegaConf.load('configs/config.yml')
    cfg.imagenet_data_module.args.mode = 'unsupervised'
    model = instantiate(cfg.simclr)
    data_module = instantiate(cfg.imagenet_data_module)
    data_module.setup('fit')
    trainer = pl.Trainer()
    trainer.fit(model, data_module)

def train_cls_imagenet():
    cfg = OmegaConf.load('configs/config.yml')
    cfg.imagenet_data_module.args.mode = 'supervised'
    cfg.imagenet_data_module.args.batch_size = 128
    cfg.classifier.args.lr = 0.3
    cfg.classifier.args.max_epochs = 200
    model = instantiate(cfg.classifier)
    data_module = instantiate(cfg.imagenet_data_module)
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_cls_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="imagenet_ckpts", save_top_k=1, save_last=True,
                                          monitor="val_acc_cls", mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    n_devices = 2
    accelerator = 'gpu'
    accumulate_grad_batches = 1
    strategy = DDPStrategy(find_unused_parameters=True, process_group_backend='gloo')
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    trainer = pl.Trainer(devices=n_devices, accelerator=accelerator, strategy=strategy,
                          accumulate_grad_batches=accumulate_grad_batches,
                          max_epochs = cfg.classifier.args.max_epochs,
                          logger = logger,
                          callbacks=[lr_monitor_callback, checkpoint_callback],
                          sync_batchnorm=True)
    trainer.fit(model, data_module)
    
def train_simclr_minst():
    cfg = OmegaConf.load('configs/config_mnist.yml')
    cfg.cifar_data_module.args.mode = 'unsupervised'
    cfg.simclr.args.lr = 1.2
    cfg.simclr.args.max_epochs = 800
    model = instantiate(cfg.simclr)
    data_module = instantiate(cfg.mnist_data_module)
    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_simclr_' +'{epoch:02d}_{train_loss_ssl:.4f}',
                                          dirpath="mnist_ckpts", save_top_k=1, save_last=True,
                                          monitor="train_loss_ssl", mode='min')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    n_devices = 2
    accelerator = 'gpu'
    accumulate_grad_batches = 2
    strategy = DDPStrategy(find_unused_parameters=True, process_group_backend='gloo')
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    trainer = pl.Trainer(devices=n_devices, accelerator=accelerator, strategy=strategy,
                          accumulate_grad_batches=accumulate_grad_batches,
                          max_epochs = cfg.simclr.args.max_epochs,
                          logger = logger,
                          callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)

def train_cls_mnist():
    cfg = OmegaConf.load('configs/config_mnist.yml')
    cfg.mnist_data_module.args.mode = 'supervised'
    cfg.classifier.args.lr = 0.3
    cfg.classifier.args.max_epochs = 200
    model = instantiate(cfg.classifier)
    data_module = instantiate(cfg.mnist_data_module)

    checkpoint_callback = ModelCheckpoint(filename=cfg.exp_name + '_cls_' +'{epoch:02d}_{val_acc_cls:.4f}',
                                          dirpath="mnist_ckpts", save_top_k=1, save_last=True,
                                          monitor="val_acc_cls", mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    # n_devices = 1
    # accelerator = 'gpu'
    # accumulate_grad_batches = 1
    # strategy = DDPStrategy(find_unused_parameters=True, process_group_backend='gloo')
    logger = TensorBoardLogger(save_dir=f'{cfg.exp_name}_log')

    # trainer = pl.Trainer(devices=n_devices, accelerator=accelerator, strategy=strategy,
    #                       accumulate_grad_batches=accumulate_grad_batches,
    #                       max_epochs = cfg.classifier.args.max_epochs,
    #                       logger = logger,
    #                       callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer = pl.Trainer(max_epochs = cfg.classifier.args.max_epochs, 
                         accelerator = 'gpu',
                         logger = logger,
                         callbacks=[lr_monitor_callback, checkpoint_callback])
    trainer.fit(model, data_module)

if __name__ == '__main__':
    pl.seed_everything(42)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # train_simclr_imagenet()
    # train_simclr_cifar()
    # train_cls_cifar()
    train_cls_mnist()
