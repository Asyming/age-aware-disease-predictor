import os
import torch
import numpy as np
from torchmetrics import AUROC, AveragePrecision
from src.teacher_models import *
from src.utils import GradientQueue, gradient_clipping, mixup

class Trainer:
    def __init__(self, model, criterion, optimizer, device, model_name, save_dir, snp_ids=None, batched_g=None, gene_g=None, norm_weight=1.0, args=None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.norm_weight = norm_weight
        self.snp_ids = torch.tensor(snp_ids).to(device) if snp_ids is not None else None
        self.batched_g = batched_g.to(device) if batched_g is not None else None
        self.gene_g = gene_g.to(device) if gene_g is not None else None
        self.auroc = AUROC(task='binary').to(device)
        self.ap = AveragePrecision(task='binary').to(device)
        self.args = args
        self.age_threshold = getattr(args, 'age_threshold', 65)
        self.use_label_correction = getattr(args, 'use_label_correction', False)
        self.eval_interval = getattr(args, 'eval_interval', 10)
        self.n_steps = getattr(args, 'n_steps', 20000)
        self.n_early_stop = getattr(args, 'n_early_stop', 10)
        self.log_interval = getattr(args, 'log_interval', 20)
        self.pseudo_label_interval = getattr(args, 'pseudo_label_interval', 20)
        self.pseudo_label_start_step = getattr(args, 'pseudo_label_start_step', 200)
        self.eta = getattr(args, 'eta', 0.1)
        self.gradnorm_queue = GradientQueue(maxlen=32)
        for _ in range(self.gradnorm_queue.maxlen):
            self.gradnorm_queue.add(1.0)
        self.use_mixup = getattr(args, 'use_mixup', False)
        self.mixup_alpha = getattr(args, 'mixup_alpha', 0.3)

    def train(self, train_loader, val_loader, test_loader, es_window=3, train_eval_loader=None):
        print(f"----------------Training {self.model_name}----------------")
        running_loss, val_scores = [], []
        cur_early_stop, best_score, step, test_auprc = 0, 0, 0, 0.0

        for inputs, labels, ages in train_loader:
            self.model.train()
            step += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            ages = ages.to(self.device)

            if self.use_label_correction and step >= self.pseudo_label_start_step and step % self.pseudo_label_interval == 0:
                with torch.no_grad():
                    self.model.eval()
                    if isinstance(self.model, (AgeAwareMLP1, AgeAwareMLP2)):
                        logits, _, _ = self.model(inputs, ages, labels)
                    elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                         logits, _, _ = self.model(inputs, self.snp_ids, self.batched_g, ages, labels)
                    elif isinstance(self.model, ctrMLP):
                         logits, _ = self.model(inputs, ages=ages, labels=labels)
                    elif isinstance(self.model, UGP_v1):
                        logits, _ = self.model(inputs, self.snp_ids, self.batched_g)
                    elif isinstance(self.model, UGP_v2):
                        linear_preds, nonlinear_preds, _ = self.model(inputs, self.snp_ids, self.batched_g)
                        logits = linear_preds + nonlinear_preds
                    elif isinstance(self.model, UGP_v3):
                         logits, _, _ = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                    elif isinstance(self.model, ctrUGP_v1):
                         logits, _, _, _ = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g, ages=ages, labels=labels)
                    elif isinstance(self.model, MLP):
                        logits = self.model(inputs)

                    probs = torch.sigmoid(logits).squeeze()
                    self.model.train()

                pos_probs = probs[labels.squeeze() == 1]
                confidence = pos_probs.mean().item()
                print(f"[{step}] confidence: {confidence:.4f}")

                original_labels_squeezed = labels.squeeze().detach()
                pseudo_labels = original_labels_squeezed.clone()
                condition_mask = (ages.squeeze() <= self.age_threshold) & (original_labels_squeezed == 0)
                confident_positive_mask = probs > confidence
                update_indices = torch.where(condition_mask & confident_positive_mask)[0]

                low_neg_probs = probs[condition_mask]
                n_upper = (low_neg_probs > 0.5).sum().item()
                max_low_neg = low_neg_probs.max().item()
                print(f"[{step}] n_upper(>0.5): {n_upper}/{low_neg_probs.size(0)}, max_low_neg_prob: {max_low_neg:.4f}")

                if len(update_indices) > 0:
                    pseudo_labels[update_indices] = self.eta * probs[update_indices]
                    labels = pseudo_labels.unsqueeze(1)
                    print(f"Corrected {len(update_indices)}/{low_neg_probs.size(0)}.")

            self.optimizer.zero_grad()
            if self.use_mixup:
                mixup_inputs, mixup_labels, _ = mixup(inputs, labels, self.mixup_alpha)
                if isinstance(self.model, (AgeAwareMLP1, AgeAwareMLP2)):
                    outputs, _, norm_loss = self.model(mixup_inputs, ages, mixup_labels)
                    loss = self.criterion(outputs, mixup_labels) + self.norm_weight * norm_loss
                elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                    outputs, _, norm_loss = self.model(mixup_inputs, self.snp_ids, self.batched_g, ages, mixup_labels)
                    loss = self.criterion(outputs, mixup_labels) + self.norm_weight * norm_loss
                elif isinstance(self.model, ctrMLP):
                    outputs, norm_loss = self.model(mixup_inputs, ages=ages, labels = mixup_labels)
                    loss = self.criterion(outputs, mixup_labels) + self.norm_weight * norm_loss
                elif isinstance(self.model, UGP_v1):
                    outputs, _ = self.model(mixup_inputs, self.snp_ids, self.batched_g)
                    loss = self.criterion(outputs, mixup_labels)
                elif isinstance(self.model, UGP_v2):
                    linear_preds, nonlinear_preds, _ = self.model(mixup_inputs, self.snp_ids, self.batched_g)
                    outputs = linear_preds + nonlinear_preds
                    loss1 = self.criterion(linear_preds, mixup_labels)
                    loss2 = self.criterion(nonlinear_preds, mixup_labels)
                    loss3 = self.criterion(outputs, mixup_labels)
                    loss = loss1 + loss2 + loss3
                elif isinstance(self.model, UGP_v3):
                    outputs, _, _ = self.model(mixup_inputs, self.snp_ids, self.batched_g, self.gene_g)
                    loss = self.criterion(outputs, mixup_labels)
                elif isinstance(self.model, ctrUGP_v1):
                    outputs, _, _, norm_loss = self.model(mixup_inputs, self.snp_ids, self.batched_g, self.gene_g, ages=ages, labels=mixup_labels)
                    loss = self.criterion(outputs, mixup_labels) + self.norm_weight * norm_loss
                elif isinstance(self.model, MLP): # Basic MLP
                    outputs = self.model(mixup_inputs)
                    loss = self.criterion(outputs, mixup_labels)
            else:
                if isinstance(self.model, (AgeAwareMLP1, AgeAwareMLP2)):
                    outputs, _, norm_loss = self.model(inputs, ages, labels)
                    loss = self.criterion(outputs, labels) + self.norm_weight * norm_loss
                elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                    outputs, _, norm_loss = self.model(inputs, self.snp_ids, self.batched_g, ages, labels)
                    loss = self.criterion(outputs, labels) + self.norm_weight * norm_loss
                elif isinstance(self.model, ctrMLP):
                    outputs, norm_loss = self.model(inputs, ages=ages, labels = labels)
                    loss = self.criterion(outputs, labels) + self.norm_weight * norm_loss
                elif isinstance(self.model, UGP_v1):
                    outputs, _ = self.model(inputs, self.snp_ids, self.batched_g)
                    loss = self.criterion(outputs, labels)
                elif isinstance(self.model, UGP_v2):
                    linear_preds, nonlinear_preds, _ = self.model(inputs, self.snp_ids, self.batched_g)
                    outputs = linear_preds + nonlinear_preds
                    loss1 = self.criterion(linear_preds, labels)
                    loss2 = self.criterion(nonlinear_preds, labels)
                    loss3 = self.criterion(outputs, labels)
                    loss = loss1 + loss2 + loss3
                elif isinstance(self.model, UGP_v3):
                    outputs, _, _ = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                    loss = self.criterion(outputs, labels)
                elif isinstance(self.model, ctrUGP_v1):
                    outputs, _, _, norm_loss = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g, ages=ages, labels=labels)
                    loss = self.criterion(outputs, labels) + self.norm_weight * norm_loss
                elif isinstance(self.model, MLP): # Basic MLP
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)                

            loss.backward()
            gradient_clipping(self.model, self.gradnorm_queue)
            self.optimizer.step()
            running_loss.append(loss.item())

            if (step + 1) % self.log_interval == 0:
                log_msg = f"[{step + 1}] loss: {np.mean(running_loss):.3f}"
                print(log_msg, flush=True)
                running_loss = []

            if (step + 1) % self.eval_interval == 0:
                print("----------------Validating----------------", flush=True)
                val_metrics = self.evaluate(val_loader)
                val_auprc_score = val_metrics['auprc']
                val_scores.append(val_auprc_score)
                if len(val_scores) > es_window:
                    val_scores.pop(0)
                avg_score = np.mean(val_scores)
                
                improved = False
                if avg_score > best_score:
                    best_score = avg_score
                    improved = True
                    cur_early_stop = 0
                elif avg_score <= best_score and val_auprc_score > best_score:
                    best_score = val_auprc_score
                    improved = True
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1

                if improved:
                    test_metrics = self.evaluate(test_loader)
                    test_auprc = test_metrics['auprc']
                    save_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'test_metrics': test_metrics,
                        'val_avg_auprc': best_score
                    }
                    torch.save(save_dict, os.path.join(self.save_dir, f'{self.model_name}.pth'))

                if cur_early_stop >= self.n_early_stop:
                    print(f'\nEarly stopping triggered after {self.n_early_stop} evaluations without improvement.')
                    print(f'Best val AUPRC: {best_score:.5f} Best test AUPRC: {test_auprc:.5f}')
                    break

                print(f"[{step+1}] val_auprc: {val_auprc_score:.5f}, avg_score: {avg_score:.5f}, best_score: {best_score:.5f}, test_auprc: {test_auprc:.5f}, es_counter: {cur_early_stop}/{self.n_early_stop}", flush=True)
                print("----------------Training----------------", flush=True)
                self.model.train()

            if step >= self.n_steps:
                 break

        best_checkpoint_path = os.path.join(self.save_dir, f'{self.model_name}.pth')
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)

        return checkpoint

    def evaluate(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            all_logits, all_labels = [], []

            for inputs, labels, ages in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).to(torch.long)
                ages = ages.to(self.device)
                logits = None

                if isinstance(self.model, (MLP, AgeAwareMLP1, AgeAwareMLP2)):
                    logits = self.model(inputs)
                elif isinstance(self.model, ctrMLP):
                    logits, _ = self.model(inputs, ages=ages)
                elif isinstance(self.model, UGP_v1):
                    logits, _ = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g)
                elif isinstance(self.model, UGP_v2):
                    linear_preds, nonlinear_preds, _ = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g)
                    logits = linear_preds + nonlinear_preds
                elif isinstance(self.model, UGP_v3):
                    logits, _, _ = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g, gene_g=self.gene_g)
                elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                    logits = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g, ages=ages)
                elif isinstance(self.model, ctrUGP_v1):
                    logits, _, _, _ = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g, gene_g=self.gene_g, ages=ages)

                if logits is not None:
                    all_logits.append(logits.detach())
                    all_labels.append(labels.detach())

            logits = torch.cat(all_logits).reshape(-1)
            labels = torch.cat(all_labels).reshape(-1)

            results = {
                'auroc': self.auroc(logits, labels).item(),
                'auprc': self.ap(logits, labels).item(),
                'predictions': logits.cpu(),
                'labels': labels.cpu()
            }
        return results

class KDTrainer:
    def __init__(self, teacher_model, student_model, criterion, optimizer, device, model_name, save_dir, teacher_model_path, snp_ids=None, batched_g=None, gene_g=None, norm_weight=1.0, args=None):
        self.teacher_model = teacher_model
        checkpoint = torch.load(teacher_model_path, map_location=device)
        self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
        self.teacher_model = self.teacher_model.to(device)
        print(f"Loaded teacher model from: {teacher_model_path}")   
        self.model = student_model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.snp_ids = torch.tensor(snp_ids).to(device) if snp_ids is not None else None
        self.batched_g = batched_g.to(device) if batched_g is not None else None
        self.gene_g = gene_g.to(device) if gene_g is not None else None
        self.auroc = AUROC(task='binary').to(device)
        self.ap = AveragePrecision(task='binary').to(device)
        self.args = args
        self.age_threshold = getattr(args, 'age_threshold', 65)
        self.eval_interval = getattr(args, 'eval_interval', 10)
        self.n_steps = getattr(args, 'n_steps', 20000)
        self.n_early_stop = getattr(args, 'n_early_stop', 10)
        self.log_interval = getattr(args, 'log_interval', 20)
        self.use_label_correction = getattr(args, 'use_label_correction', False)
        self.pseudo_label_interval = getattr(args, 'pseudo_label_interval', 20)
        self.pseudo_label_start_step = getattr(args, 'pseudo_label_start_step', 200)
        self.confidence_threshold = getattr(args, 'confidence_threshold', 0.60)
        self.norm_weight = norm_weight
        self.gradnorm_queue = GradientQueue(maxlen=32)
        for _ in range(self.gradnorm_queue.maxlen):
            self.gradnorm_queue.add(1.0)

    def train(self, train_loader, val_loader, test_loader, es_window=3, train_eval_loader=None):
        print("----------------Training student model with best teacher----------------")
        running_loss, val_scores = [], []
        cur_early_stop, best_score, step, test_auprc = 0, 0, 0, 0.0

        for inputs, labels, ages in train_loader:
            self.model.train()
            step += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            ages = ages.to(self.device)

            with torch.no_grad():
                 if isinstance(self.teacher_model, (UGP_v1)):
                     teacher_logits, _ = self.teacher_model(inputs, self.snp_ids, self.batched_g)
                 elif isinstance(self.teacher_model, (UGP_v2)):
                     l, nl, _ = self.teacher_model(inputs, self.snp_ids, self.batched_g)
                     teacher_logits = l + nl
                 elif isinstance(self.teacher_model, UGP_v3):
                     teacher_logits, _, _ = self.teacher_model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                 elif isinstance(self.teacher_model, (AgeAwareMLP1, AgeAwareMLP2)):
                     teacher_logits, _, _ = self.teacher_model(inputs, ages, labels)
                 elif isinstance(self.teacher_model, (AgeUGP_v1, AgeUGP_v2)):
                     teacher_logits, _, _ = self.teacher_model(inputs, self.snp_ids, self.batched_g, ages, labels)
                 elif isinstance(self.teacher_model, ctrMLP):
                     teacher_logits, _ = self.teacher_model(inputs, ages=ages, labels=labels)
                 elif isinstance(self.teacher_model, ctrUGP_v1):
                     teacher_logits, _, _, _ = self.teacher_model(inputs, self.snp_ids, self.batched_g, self.gene_g, ages=ages, labels=labels)
                 elif isinstance(self.teacher_model, MLP):
                     teacher_logits = self.teacher_model(inputs)

            if self.use_label_correction and step >= self.pseudo_label_start_step and step % self.pseudo_label_interval == 0:
                with torch.no_grad():
                    self.model.eval()
                    if isinstance(self.model, UGP_v1):
                        student_logits_for_lc, _ = self.model(inputs, self.snp_ids, self.batched_g)
                    elif isinstance(self.model, UGP_v2):
                        linear_preds, nonlinear_preds, _ = self.model(inputs, self.snp_ids, self.batched_g)
                        student_logits_for_lc = linear_preds + nonlinear_preds
                    elif isinstance(self.model, (AgeAwareMLP1, AgeAwareMLP2)):
                         student_logits_for_lc, _, _ = self.model(inputs, ages, labels)
                    elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                         student_logits_for_lc, _, _ = self.model(inputs, self.snp_ids, self.batched_g, ages, labels)
                    elif isinstance(self.model, UGP_v3):
                         student_logits_for_lc, _, _ = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                    elif isinstance(self.model, ctrMLP):
                         student_logits_for_lc, _ = self.model(inputs, ages=ages, labels=labels)
                    elif isinstance(self.model, ctrUGP_v1):
                         student_logits_for_lc, _, _, _ = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g, ages=ages, labels=labels)
                    elif isinstance(self.model, MLP):
                        student_logits_for_lc = self.model(inputs)

                    probs = torch.sigmoid(student_logits_for_lc).squeeze()

                    self.model.train()

                original_labels_squeezed = labels.squeeze().detach()
                pseudo_labels = original_labels_squeezed.clone()
                condition_mask = (ages.squeeze() <= self.age_threshold) & (original_labels_squeezed == 0)
                confident_positive_mask = probs > 100000###改
                update_indices = torch.where(condition_mask & confident_positive_mask)[0]

                low_neg_probs = probs[condition_mask]
                if low_neg_probs.size(0) > 0:
                    n_upper = (low_neg_probs > 0.5).sum().item()
                    max_low_neg = low_neg_probs.max().item()
                    print(f"[{step}] n_upper(>0.5): {n_upper}/{low_neg_probs.size(0)}, max_low_neg_prob: {max_low_neg:.4f}")

                if len(update_indices) > 0:
                    pseudo_labels[update_indices] = 1
                    print(f"Corrected {len(update_indices)}/{low_neg_probs.size(0)}.")

                labels = pseudo_labels.unsqueeze(1)

            self.optimizer.zero_grad()

            if isinstance(self.model, UGP_v1):
                student_logits, _ = self.model(inputs, self.snp_ids, self.batched_g)
                loss = self.criterion(student_logits, labels, teacher_logits)
            elif isinstance(self.model, UGP_v2):
                linear_preds, nonlinear_preds, _ = self.model(inputs, self.snp_ids, self.batched_g)
                student_logits = linear_preds + nonlinear_preds
                loss1 = self.criterion(linear_preds, labels, teacher_logits)
                loss2 = self.criterion(nonlinear_preds, labels, teacher_logits)
                loss3 = self.criterion(student_logits, labels, teacher_logits)
                loss = loss1 + loss2 + loss3
            elif isinstance(self.model, (AgeAwareMLP1, AgeAwareMLP2)):
                student_logits, _, norm_loss = self.model(inputs, ages, labels)
                loss = self.criterion(student_logits, labels, teacher_logits) + self.norm_weight * norm_loss
            elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                student_logits, _, norm_loss = self.model(inputs, self.snp_ids, self.batched_g, ages, labels)
                loss = self.criterion(student_logits, labels, teacher_logits) + self.norm_weight * norm_loss
            elif isinstance(self.model, UGP_v3):
                student_logits, _, _ = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                loss = self.criterion(student_logits, labels, teacher_logits)
            elif isinstance(self.model, ctrMLP):
                 student_logits, norm_loss = self.model(inputs, teacher_logits=teacher_logits, ages=ages, labels=labels)
                 loss = self.criterion(student_logits, labels, teacher_logits) + self.norm_weight * norm_loss
            elif isinstance(self.model, ctrUGP_v1):
                 student_logits, _, _, norm_loss = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g, ages=ages, labels=labels)
                 loss = self.criterion(student_logits, labels, teacher_logits) + self.norm_weight * norm_loss
            elif isinstance(self.model, MLP):
                student_logits = self.model(inputs)
                loss = self.criterion(student_logits, labels, teacher_logits)

            loss.backward()
            gradient_clipping(self.model, self.gradnorm_queue)
            self.optimizer.step()
            running_loss.append(loss.item())
            if (step + 1) % self.log_interval == 0:
                log_msg = f"[{step + 1}] loss: {np.mean(running_loss):.3f}"
                print(log_msg, flush=True)
                running_loss = []

            if (step + 1) % self.eval_interval == 0:
                print("----------------Validating Student----------------", flush=True)
                val_metrics = self.evaluate(val_loader)
                val_auprc_score = val_metrics['auprc']
                val_scores.append(val_auprc_score)
                if len(val_scores) > es_window:
                    val_scores.pop(0)
                avg_score = np.mean(val_scores)

                improved = False
                if avg_score > best_score:
                    best_score = avg_score
                    improved = True
                    cur_early_stop = 0
                elif avg_score <= best_score and val_auprc_score > best_score:
                    best_score = val_auprc_score
                    improved = True
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1

                if improved:
                    test_metrics = self.evaluate(test_loader)
                    test_auprc = test_metrics['auprc']
                    save_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'test_metrics': test_metrics,
                        'val_avg_auprc': best_score
                    }
                    torch.save(save_dict, os.path.join(self.save_dir, f'{self.model_name}.pth'))

                if cur_early_stop >= self.n_early_stop:
                    print(f'\nEarly stopping triggered for student after {self.n_early_stop} evaluations without improvement.')
                    print(f'Best student val AUPRC: {best_score:.5f}, student test AUPRC: {test_auprc:.5f}')
                    break

                print(f"[{step+1}] val_auprc: {val_auprc_score:.5f}, avg_score: {avg_score:.5f}, best_score: {best_score:.5f}, test_auprc: {test_auprc:.5f}, es_counter: {cur_early_stop}/{self.n_early_stop}", flush=True)
                print("----------------Training Student----------------", flush=True)
                self.model.train()

            if step >= self.n_steps:
                break

        best_checkpoint_path = os.path.join(self.save_dir, f'{self.model_name}.pth')
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
        return checkpoint

    def evaluate(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            all_logits, all_labels = [], []

            for inputs, labels, ages in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).to(torch.long)
                ages = ages.to(self.device)
                logits = None

                if isinstance(self.model, (MLP, AgeAwareMLP1, AgeAwareMLP2)):
                    logits = self.model(inputs)
                elif isinstance(self.model, ctrMLP):
                    logits, _ = self.model(inputs, ages = ages)
                elif isinstance(self.model, UGP_v1):
                    logits, _ = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g)
                elif isinstance(self.model, UGP_v2):
                    linear_preds, nonlinear_preds, _ = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g)
                    logits = linear_preds + nonlinear_preds
                elif isinstance(self.model, UGP_v3):
                    logits, _, _ = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g, gene_g=self.gene_g)
                elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                    logits = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g, ages=ages)
                elif isinstance(self.model, ctrUGP_v1):
                    logits, _, _, _ = self.model(inputs, snp_ids=self.snp_ids, batched_g=self.batched_g, gene_g=self.gene_g, ages=ages)

                if logits is not None:
                    all_logits.append(logits.detach())
                    all_labels.append(labels.detach())

            logits = torch.cat(all_logits).reshape(-1)
            labels = torch.cat(all_labels).reshape(-1)

        return {
            'auroc': self.auroc(logits, labels).item(),
            'auprc': self.ap(logits, labels).item(),
            'predictions': logits.cpu(),
            'labels': labels.cpu()
        }