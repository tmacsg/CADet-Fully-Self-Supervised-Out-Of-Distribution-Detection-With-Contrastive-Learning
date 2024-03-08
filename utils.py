import os
import xmltodict
import shutil
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE
import matplotlib.pyplot as plt
import torch
import torchattacks
from utils.data_utils import AttackDataset, DatasetAttacker, DatasetAttacker_NoResize
from utils.ckpt_utils import *
from torchvision.models import resnet50,  ResNet50_Weights
from models.resnet import ResNet50
from PIL import Image
import random

def prepare_imagenet_val_set():
    """ Copy images from imagenet val folder to another folder with class names as the subfolder name
    
    """
    imagenet_val_source_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val_raw'
    imagenet_val_target_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
    imagenet_val_annot_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val'
    paths = os.listdir(imagenet_val_source_dir)
    for _, path in enumerate(tqdm(paths)):
        image_name = path.split('.')[0]
        image_annot_path = os.path.join(imagenet_val_annot_dir, image_name + '.xml')
        image_annot = xmltodict.parse(open(image_annot_path, 'rb'))
        objs = image_annot['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
        image_target_dir = os.path.join(imagenet_val_target_dir, objs[0]['name'])
        os.makedirs(image_target_dir, exist_ok=True)
        shutil.copyfile(os.path.join(imagenet_val_source_dir, path), 
                        os.path.join(image_target_dir, path))


def prepare_imagenet_pgd(num_samples=None):
    imagenet_val_source_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    cfg = OmegaConf.load("configs/config.yml")
    model = instantiate(cfg.classifier)
    model._load_weights()
    model = model.to(device)
    model.eval()
    data_attacker = DatasetAttacker(image_path=imagenet_val_source_dir, 
                                    target_path='image_data/PGD',
                                    device=device,
                                    attacker=torchattacks.PGD(model, eps=0.02, alpha=0.002, steps=50, random_start=True))
    data_attacker.attack(num_samples)

def prepare_imagenet_pgd_no_resize(num_samples=None):
    imagenet_val_source_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    cfg = OmegaConf.load("configs/config.yml")   
    model = instantiate(cfg.classifier)
    model._load_weights()
    model = model.to(device)
    model.eval()
    data_attacker = DatasetAttacker_NoResize(image_path=imagenet_val_source_dir, 
                                    target_path='image_data/PGD',
                                    device=device,
                                    attacker=torchattacks.PGD(model, eps=0.02, alpha=0.002, steps=50, random_start=True))
    data_attacker.attack(num_samples)


def prepare_imagenet_cw(num_samples=None):
    imagenet_val_source_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    cfg = OmegaConf.load("configs/config.yml")   
    model = instantiate(cfg.classifier)
    model._load_weights()
    model = model.to(device)
    model.eval()
    data_attacker = DatasetAttacker(image_path=imagenet_val_source_dir, 
                                    target_path='image_data/CW',
                                    device=device,
                                    attacker=torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.03))
    data_attacker.attack(num_samples)

def prepare_imagenet_cw_no_resize(num_samples=None):
    imagenet_val_source_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    cfg = OmegaConf.load("configs/config.yml")   
    model = instantiate(cfg.classifier)
    model._load_weights()
    model = model.to(device)
    model.eval()
    data_attacker = DatasetAttacker_NoResize(image_path=imagenet_val_source_dir, 
                                    target_path='image_data/CW',
                                    device=device,
                                    attacker=torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.03))
    data_attacker.attack(num_samples)

def prepare_imagenet_fgsm(num_samples=None):
    imagenet_val_source_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    cfg = OmegaConf.load("configs/config.yml")   
    model = instantiate(cfg.classifier)
    model._load_weights()
    model = model.to(device)
    model.eval()
    data_attacker = DatasetAttacker(image_path=imagenet_val_source_dir, 
                                    target_path='image_data/FGSM',
                                    device=device,
                                    attacker=torchattacks.FGSM(model, eps=0.05))
    data_attacker.attack(num_samples)

def prepare_imagenet_fgsm_no_resize(num_samples=None):
    imagenet_val_source_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    cfg = OmegaConf.load("configs/config.yml")   
    model = instantiate(cfg.classifier)
    model._load_weights()
    model = model.to(device)
    model.eval()
    data_attacker = DatasetAttacker_NoResize(image_path=imagenet_val_source_dir, 
                                    target_path='image_data/FGSM',
                                    device=device,
                                    attacker=torchattacks.FGSM(model, eps=0.05))
    data_attacker.attack(num_samples)


def show_difference():
    """ show difference between attacked image and original image
    """
    imagenet_val_source_dir = 'image_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
    ATTACK_TYPES = ['CW', 'FGSM', 'PGD']
    def get_random_image_path(attack_type):
        assert attack_type in ATTACK_TYPES
        dir = os.path.join('image_data',attack_type)
        sub_dir = random.choice(list(os.listdir(dir)))
        image_name = os.listdir(os.path.join(dir, sub_dir))[0]
        image_path  = os.path.join(sub_dir, image_name)
        original_image_path = os.path.join(imagenet_val_source_dir, image_path)
        attacked_image_path = os.path.join(dir, image_path)
        return original_image_path, attacked_image_path
    
    _, axes = plt.subplots(len(ATTACK_TYPES),3, figsize=(8,8))
    for i in range(len(ATTACK_TYPES)):
        original_image_path, attacked_image_path = get_random_image_path(ATTACK_TYPES[i])
        original_image = Image.open(original_image_path)
        attacked_image = Image.open(attacked_image_path)
        diff = np.abs(np.array(attacked_image, dtype=np.int8) - np.array(original_image, dtype=np.int8)) * 20
        # diff_image = Image.fromarray(diff.astype(np.uint8))
        axes[i,0].imshow(original_image)
        axes[i,0].axis('off')
        axes[i,0].set_title('ORIGINAL')
        axes[i,1].imshow(attacked_image)
        axes[i,1].axis('off')
        axes[i,1].set_title(f'{ATTACK_TYPES[i]}_ATTACKED')
        axes[i,2].imshow(diff, cmap='hot')
        axes[i,2].axis('off')
        axes[i,2].set_title('DIFFERENCE * 20')
    plt.tight_layout()
    plt.show()

def download_models():
    model = 'r50_1x_sk0'
    # path = 'downloads/' + 'r50_1x_sk0_simclr'
    path = 'downloads/' + 'r50_1x_sk0_supervised'
    os.makedirs(path, exist_ok=True)
    # simclr_category = 'pretrained'
    simclr_category = 'supervised'
    model_category = simclr_categories[simclr_category]  
    url = simclr_base_url.format(model=model, category=simclr_category)
    for file in tqdm(files):
        f = file.format(category=model_category)
        download(url + f, os.path.join(path, f))   

def extract_simclr():
    model_save_path = 'downloads/resnet-50_simclr.ckpt' 
    cfg = OmegaConf.load("configs/config.yml")  
    model = instantiate(cfg.simclr)
    weight_path = 'downloads/r50_1x_sk0_simclr'  
    extract_simclr_weights(model, weight_path, model_save_path)

def extract_classifier():
    model_save_path = 'downloads/resnet-50_classifier.ckpt' 
    cfg = OmegaConf.load("configs/config.yml")  
    model = instantiate(cfg.classifier)
    weight_path = 'downloads/r50_1x_sk0_supervised'
    extract_classifier_weights(model, weight_path, model_save_path)


if __name__ == '__main__':
    # prepare_imagenet_val_set()
    # prepare_imagenet_pgd(1000)
    # prepare_imagenet_pgd_no_resize(1000)
    # prepare_imagenet_cw(1000)
    # prepare_imagenet_cw_no_resize(1000)
    # prepare_imagenet_fgsm(1000)
    # prepare_imagenet_fgsm_no_resize(1000)
    show_difference()
    # download_models()
    # extract_simclr()
    # extract_classifier()
    