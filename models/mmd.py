import pytorch_lightning as pl
import torch
from utils.stat_utils import mmd_permutation_test, mmd_permutation_test_cc
import pandas as pd
import matplotlib.pyplot as plt
import os
from lightly.models.utils import deactivate_requires_grad
from sklearn import metrics

class MMD(pl.LightningModule):
    def __init__(self, base_model, args):
        super().__init__()     

        self.backbone = self._extract_backbone(base_model)
        deactivate_requires_grad(self.backbone)
        self.n_tests = args.n_tests
        self.n_perms = args.n_perms
        self.sample_sizes = args.sample_sizes
        self.sig_level = args.sig_level
        self.clean_calib = args.clean_calib

        self.hidden_dim = self.backbone.hidden_dim
        self.p_value_outputs = {}
        self.mmd_value_outputs = {}
        
    def on_test_start(self):
        mmd_columns = [f'test_step_{i}' for i in range(self.n_tests)]
        mmd_indexes = [f'perm_{i}' for i in range(self.n_perms)]
        for sample_size in self.sample_sizes:
            self.p_value_outputs[f'sample_size{sample_size}_same_dist'] = pd.DataFrame(columns=['test_step', 'p_value', 'threshold', 'estimation'])
            self.p_value_outputs[f'sample_size{sample_size}_diff_dist'] = pd.DataFrame(columns=['test_step', 'p_value', 'threshold', 'estimation'])            
            self.mmd_value_outputs[f'sample_size{sample_size}_same_dist'] = pd.DataFrame(columns=mmd_columns, index=mmd_indexes)
            self.mmd_value_outputs[f'sample_size{sample_size}_diff_dist'] = pd.DataFrame(columns=mmd_columns, index=mmd_indexes)
    
    def test_step(self, batch, batch_idx):
        sample_s, _ = batch['s']
        sample_q, _ = batch['q']

        # import matplotlib.pyplot as plt
        # _, axes = plt.subplots(2, len(sample_q), figsize=(len(sample_q), 4))
        # for i in range(len(sample_q)):
        #     axes[0,i].imshow(sample_s[i].cpu().permute(1, 2, 0))
        #     axes[0,i].axis('off')
        #     axes[0, i].set_title('Samples S', fontsize=10)
        #     axes[1,i].imshow(sample_q[i].cpu().permute(1, 2, 0))
        #     axes[1,i].axis('off')
        #     axes[1, i].set_title('Samples Q', fontsize=10)
        # plt.tight_layout()
        # plt.show()

        feature_s = self.backbone(sample_s).flatten(start_dim=1)
        feature_q = self.backbone(sample_q).flatten(start_dim=1)
        index_s = torch.randperm(len(feature_s))
        index_q = torch.randperm(len(feature_q))
        for sample_size in self.sample_sizes:
            feature_s1 = feature_s[index_s[:sample_size]]
            feature_s2 = feature_s[index_s[sample_size:sample_size*2]]
            feature_s3 = feature_s[index_s[sample_size*2:sample_size*3]]     
            feature_q1 = feature_q[index_q[:sample_size]]     
            if self.clean_calib:
                p_value_diff_dist, mmd_vals_diff_dist, est_diff_dist = mmd_permutation_test_cc(feature_s1, 
                                                                                            feature_s2, 
                                                                                            feature_q1, 
                                                                                            self.n_perms)
                p_value_same_dist, mmd_vals_same_dist, est_same_dist = mmd_permutation_test_cc(feature_s1, 
                                                                                            feature_s2, 
                                                                                            feature_s3, 
                                                                                            self.n_perms)
            else:
                p_value_diff_dist, mmd_vals_diff_dist, est_diff_dist = mmd_permutation_test(feature_s1, 
                                                                                            feature_q1, 
                                                                                            self.n_perms)
                p_value_same_dist, mmd_vals_same_dist, est_same_dist = mmd_permutation_test(feature_s1, 
                                                                                            feature_s3, 
                                                                                            self.n_perms)
            threshold_diff_dist = torch.quantile(mmd_vals_diff_dist, 1 - self.sig_level).item()
            threshold_same_dist = torch.quantile(mmd_vals_same_dist, 1 - self.sig_level).item()
            self.p_value_outputs[f'sample_size{sample_size}_same_dist'].loc[batch_idx] = [batch_idx, p_value_same_dist, threshold_same_dist, est_same_dist]
            self.p_value_outputs[f'sample_size{sample_size}_diff_dist'].loc[batch_idx] = [batch_idx, p_value_diff_dist, threshold_diff_dist, est_diff_dist]
            self.mmd_value_outputs[f'sample_size{sample_size}_same_dist'][f'test_step_{batch_idx}'] = mmd_vals_same_dist
            self.mmd_value_outputs[f'sample_size{sample_size}_diff_dist'][f'test_step_{batch_idx}'] = mmd_vals_diff_dist

    def on_test_end(self):
        with pd.ExcelWriter(os.path.join(self.logger.log_dir, 'p_values.xlsx')) as writer1:
            for k, v in self.p_value_outputs.items():
                v.to_excel(writer1, sheet_name=k, index=False)

        with pd.ExcelWriter(os.path.join(self.logger.log_dir, 'mmd_values.xlsx')) as writer2:
            for k, v in self.mmd_value_outputs.items():
                v.to_excel(writer2, sheet_name=k)

        rejection_rate_indexes = [f'sample_size{i}_{j}' for j in ['same_dist', 'diff_dist'] for i in self.sample_sizes]
        rejection_rates = {}
        for key in rejection_rate_indexes:
            rejection_rate =  (self.p_value_outputs[key]['p_value'] < self.sig_level).sum() / float(len(self.p_value_outputs[key]))
            rejection_rates[key] = [rejection_rate]     
        aurocs = self.save_roc_curve()
        pd.DataFrame.from_dict(rejection_rates, orient='index').to_csv(os.path.join(self.logger.log_dir, 'rejection_rates.csv'), header=False)
        pd.DataFrame.from_dict(aurocs, orient='index').to_csv(os.path.join(self.logger.log_dir, 'auroc.csv'), header=False)
        
        self.p_value_outputs.clear()
        self.mmd_value_outputs.clear()

    def save_roc_curve(self):
        plt.figure(figsize=(12, 8))
        keys = [f'sample_size{i}' for i in self.sample_sizes]
        aurocs = {}
        for key in keys:
            pvalue_same_dist = self.p_value_outputs[f'{key}_same_dist']['p_value'].to_list()
            lable_same_dist = [0] * len(pvalue_same_dist)
            pvalue_diff_dist = self.p_value_outputs[f'{key}_diff_dist']['p_value'].to_list()
            lable_diff_dist = [1] * len(pvalue_diff_dist)
            pvalue_same_dist.extend(pvalue_diff_dist), lable_same_dist.extend(lable_diff_dist)
            pvalue_all = torch.tensor(pvalue_same_dist)
            lable_all = torch.tensor(lable_same_dist)
            fpr, tpr, _ = metrics.roc_curve(lable_all, [1 - p for p in pvalue_all])
            auroc = metrics.auc(fpr, tpr)         
            aurocs[key] = auroc
            plt.plot(fpr, tpr, lw=2, label=f'{key}: AUC = {auroc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for  MMD Two Sample Test')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.logger.log_dir, 'roc_curve.png'))
        return aurocs

    def _extract_backbone(self, base_model):
        base_model._load_weights()
        return base_model.backbone