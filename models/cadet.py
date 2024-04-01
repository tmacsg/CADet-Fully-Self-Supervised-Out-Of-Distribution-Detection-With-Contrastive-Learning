import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from utils.data_utils import CADetTransform
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.data_utils import ImageNetDataModule
from lightly.models.utils import deactivate_requires_grad
from sklearn import metrics
import numpy as np
from utils.stat_utils import GuassianKernel

class CADet(pl.LightningModule):
    def __init__(self, base_model, val_dataset, args):
        super().__init__()

        self.backbone = self._extract_backbone(base_model)
        deactivate_requires_grad(self.backbone)
        self.test_image_sets = args.test_image_sets
        assert self.test_image_sets[0] == 'same_dist'
        self.n_tests = args.n_tests
        self.n_transforms = args.n_transforms
        self.sample_size_1 = args.sample_size_1
        self.sample_size_2 = args.sample_size_2

        self.val_dataset = val_dataset
        self.val_split = [self.sample_size_1, self.sample_size_2, self.n_tests,
                          len(self.val_dataset) - self.sample_size_1- self.sample_size_2 - self.n_tests]

        self.gamma, self.scores, self.X_1_feats = None, None, None
        self.p_value_outputs = {}
        self.m_in = {}
        self.m_out = {}
        self.test_scores = {}
        
        self.m_value_outputs = None
        

    def on_test_start(self):
        if os.path.exists('CADet_Calib/CADet_Calib.npy'):
            print("load calibration data from file...")
            calib_data = np.load('CADet_Calib/CADet_Calib.npy', allow_pickle=True)
            self.X_1_feats = torch.tensor(calib_data.item()['features'], device=self.device)
            self.scores = torch.tensor(calib_data.item()['scores'], device=self.device)
            self.gamma = calib_data.item()['gamma']
            self.same_dist_indices =  calib_data.item()['same_dist']
        else:
            print("No calibration file found, do calibration...")
            val_dataset_1, val_dataset_2, same_dist_dataset, _ = random_split(self.val_dataset, self.val_split)
            self.same_dist_indices = np.array(same_dist_dataset.indices)
            calibrated_data = self.calibrate(val_dataset_1, val_dataset_2)
            calibrated_data['same_dist'] = self.same_dist_indices
            # os.makedirs('CADet_Calib', exist_ok=True)
            # np.save(os.path.join('CADet_Calib', 'CADet_Calib.npy'), calibrated_data)
                       
        for key in self.test_image_sets: 
            self.p_value_outputs[key] = pd.DataFrame(columns=['test_step', 'p_value'])
            self.m_in[key] = torch.zeros(self.n_tests)
            self.m_out[key] = torch.zeros(self.n_tests)
            self.test_scores[key] = torch.zeros(self.n_tests)
        self.m_value_outputs = pd.DataFrame(columns=['m_in_mean', 'm_out_mean', 'gamma_m_out', 'm_in_var', 'm_out_var'], 
                                            index=self.test_image_sets)
        
        # calculate in-distribution m values
        same_dist_dataset = Subset(self.val_dataset, self.same_dist_indices)
        same_dist_data_loader = DataLoader(dataset=same_dist_dataset, batch_size=1)
        for batch_idx, (X, _) in enumerate(same_dist_data_loader):
            X = X[0].to(self.device)
            X_test_feat = self.backbone(X)
            m_in, m_out = self.compute(X_test_feat)
            test_score = m_in + m_out * self.gamma 
            p_value = (torch.sum(test_score > self.scores).item() + 1) / (len(self.scores) + 1) 
            self.p_value_outputs['same_dist'].loc[batch_idx] = [batch_idx, p_value]
            self.m_in['same_dist'][batch_idx] = m_in
            self.m_out['same_dist'][batch_idx] = m_out
            self.test_scores['same_dist'][batch_idx] = test_score

    def test_step(self, batch, batch_idx):
        for key in self.test_image_sets[1:]:
            X_test, _ = batch[key]
            X_test = X_test[0]

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(1, len(X_test), figsize=(len(X_test), 2))
            # for i in range(len(X_test)):
            #     axes[i].imshow(X_test[i].cpu().permute(1, 2, 0))
            #     axes[i].axis('off')
            # fig.suptitle(key, fontsize=16)
            # plt.tight_layout()
            # plt.show()
            
            X_test_feat = self.backbone(X_test)
            m_in, m_out = self.compute(X_test_feat)
            test_score = m_in + m_out * self.gamma 
            p_value = (torch.sum(test_score > self.scores).item() + 1) / (len(self.scores) + 1) 
            self.p_value_outputs[key].loc[batch_idx] = [batch_idx, p_value]
            self.m_in[key][batch_idx] = m_in
            self.m_out[key][batch_idx] = m_out
            self.test_scores[key][batch_idx] = test_score

    def on_test_end(self):
        with pd.ExcelWriter(os.path.join(self.logger.log_dir, 'cadet_p_values.xlsx')) as writer:
            for k, v in self.p_value_outputs.items():
                v.to_excel(writer, sheet_name=k, index=False)
        aurocs = self.save_roc_curve()
        pd.DataFrame.from_dict(self.m_in).to_csv(os.path.join(self.logger.log_dir, 'cadet_m_in_raw.csv'))
        pd.DataFrame.from_dict(self.m_out).to_csv(os.path.join(self.logger.log_dir, 'cadet_m_out_raw.csv'))
        pd.DataFrame.from_dict(self.test_scores).to_csv(os.path.join(self.logger.log_dir, 'cadet_test_scores_raw.csv'))
        pd.DataFrame.from_dict(aurocs, orient='index').to_csv(os.path.join(self.logger.log_dir, 'cadet_aurocs.csv'), header=False)
        for key in self.test_image_sets:
            m_in = torch.mean(self.m_in[key]).item()
            m_out = torch.mean(self.m_out[key]).item()
            gamma_m_out = self.gamma * m_out
            m_in_var = torch.var(self.m_in[key]).item()
            m_out_var = torch.var(self.m_out[key]).item()           
            self.m_value_outputs.loc[key] = [m_in, m_out, gamma_m_out, m_in_var, m_out_var]             
        pd.DataFrame.from_dict(self.m_value_outputs).to_csv(os.path.join(self.logger.log_dir, 'cadet_m_values.csv'), header=True)     
        self.gamma, self.scores, self.X_1_feats = None, None, None
        self.p_value_outputs.clear()
        self.m_value_outputs = None
    
    def _cal_similarity(self, input0, input1):
        # return torch.exp(-torch.cdist(input0, input1, p=1))
        # n1 = input0.shape[0]   
        # kernels = GuassianKernel()(input0, input1)[:n1, n1:]
        # return kernels
        out0 = F.normalize(input0, dim=1)
        out1 = F.normalize(input1, dim=1)
        return out0 @ out1.t()
        # return input0 @ input1.t() 
    
    def calibrate(self, dataset_1, dataset_2):
        dataloader_1 = DataLoader(dataset_1, batch_size=1, shuffle=False)
        dataloader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False) 
        X_1_feats = []
        m_ins = []   
        m_outs = []

        for _, (X_1,_) in enumerate(tqdm(dataloader_1, desc="Compute X_VAL_1 features")):
            X_1 = X_1[0].to(self.device)
            X_1_feat = self.backbone(X_1)
            X_1_feats.append(X_1_feat)
        self.X_1_feats = torch.stack(X_1_feats)
        
        for _, (X_2,_) in enumerate(tqdm(dataloader_2, desc="Calibrate with X_VAL_2")):
            X_2 = X_2[0].to(self.device)
            X_2_feat = self.backbone(X_2)
            m_in, m_out = self.compute(X_2_feat)
            m_ins.append(m_in)
            m_outs.append(m_out)
            
        m_ins = torch.tensor(m_ins, device=self.device)
        m_outs = torch.tensor(m_outs, device=self.device)
        self.gamma = torch.sqrt(m_ins.var() / m_outs.var()).item()
        print(f'Calibrated m_in mean: ', m_ins.mean().item())
        print(f'Calibrated m_out mean: ', m_outs.mean().item())
        print(f'Calibrated gamma: {self.gamma}')
        self.scores = m_ins + m_outs * self.gamma        
        calibrated_data = {
            'features': self.X_1_feats.cpu().numpy(),
            'scores': self.scores.cpu().numpy(),
            'gamma': self.gamma
        }
        
        return calibrated_data
    
    def compute(self, X_test_feat):
        intra_sim = self._cal_similarity(X_test_feat, X_test_feat)
        m_in = (intra_sim.sum() - intra_sim.trace())  / (self.n_transforms * (self.n_transforms -1 ))     
        outer_sims = torch.vmap(lambda arg: self._cal_similarity(X_test_feat, arg))(self.X_1_feats)
        m_out = outer_sims.sum() / (self.n_transforms * self.n_transforms * self.sample_size_1)
        return m_in, m_out
    
    def save_roc_curve(self):
        assert 'same_dist' == self.test_image_sets[0]
        plt.figure(figsize=(12, 8))
        aurocs = {}
        p_values = [None] * self.n_tests * 2
        labels = [None] * self.n_tests * 2
        pvalue_same_dist = self.p_value_outputs['same_dist']['p_value'].to_list()
        p_values[:self.n_tests] = pvalue_same_dist
        labels[:self.n_tests] = [0] * self.n_tests
        labels[self.n_tests:] = [1] * self.n_tests
        labels = torch.tensor(labels)
        for key in self.test_image_sets[1:]:
            pvalue_diff_dist = self.p_value_outputs[key]['p_value'].to_list()
            p_values[self.n_tests:] = pvalue_diff_dist
            fpr, tpr, _ = metrics.roc_curve(labels, [1 - p for p in p_values])
            auroc = metrics.auc(fpr, tpr)
            aurocs[key] = auroc
            plt.plot(fpr, tpr, lw=2, label=f'{key}: AUC = {auroc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for  CADet Test')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.logger.log_dir, 'cadet_roc_curve.png'))
        return aurocs
        
    def _extract_backbone(self, base_model):
        base_model._load_weights()
        return base_model.backbone