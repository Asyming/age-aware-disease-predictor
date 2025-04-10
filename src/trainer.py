import os
import torch
import numpy as np
from torchmetrics import AUROC, AveragePrecision
from src.teacher_models import MLP, AgeAwareMLP1, AgeAwareMLP2, UGP_v1, UGP_v2, AgeUGP_v1, AgeUGP_v2, UGP_v3
from src.utils import GradientQueue, gradient_clipping

# Trainer_g for pure UGP models
# Trainer for other models
# KDTrainer for all models
class Trainer:
    def __init__(self, model, criterion, optimizer, device, model_name, save_dir, snp_ids=None, batched_g=None, norm_weight=1.0, eval_interval=10, n_steps=20000, n_early_stop=10, log_interval=20):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.norm_weight = norm_weight
        self.snp_ids = torch.tensor(snp_ids)
        self.batched_g = batched_g  
        self.auroc = AUROC(task='binary').to(device)
        self.ap = AveragePrecision(task='binary').to(device)
        self.eval_interval = eval_interval
        self.n_steps = n_steps
        self.n_early_stop = n_early_stop
        self.log_interval = log_interval
        self.gradnorm_queue = GradientQueue(maxlen=32)
        for _ in range(self.gradnorm_queue.maxlen):
            self.gradnorm_queue.add(1.0)

    def train(self, train_loader, val_loader, test_loader, es_window=3, train_eval_loader=None):
        print(f"----------------Training {self.model_name}----------------")
        running_loss, val_scores = [], []
        cur_early_stop, best_score, step = 0, 0, 0
        for inputs, labels, ages in train_loader:
            self.model.train()
            step += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            ages = ages.to(self.device)
            self.optimizer.zero_grad()
            if isinstance(self.model, (AgeAwareMLP1, AgeAwareMLP2)):
                outputs, _, norm_loss = self.model(inputs, ages, labels)
                loss = self.criterion(outputs, labels) + self.norm_weight * norm_loss
            elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                outputs, _, norm_loss = self.model(inputs, self.snp_ids, self.batched_g, ages, labels)
                loss = self.criterion(outputs, labels) + self.norm_weight * norm_loss
            else: # MLP
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            loss.backward()
            gradient_clipping(self.model, self.gradnorm_queue)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_loss.append(loss.item())
            if (step + 1) % self.log_interval == 0:
                print(f"[{step + 1}] loss: {np.mean(running_loss):.3f}", flush=True)
                running_loss = []
            if (step + 1) % self.eval_interval == 0: 
                """
                val_auprc: val_auprc at this step. 
                avg_score: average val_auprc over the last es_window. 
                best_score: best avg_score or best val_auprc util this step.
                test_auprc: test_auprc of the best model.
                """
                print("----------------Validating----------------", flush=True)
                val_metrics = self.evaluate(val_loader)
                val_auprc_score = val_metrics['auprc']
                val_scores.append(val_auprc_score)
                if len(val_scores) > es_window: 
                    val_scores.pop(0)
                avg_score = np.mean(val_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    test_metrics = self.evaluate(test_loader)
                    test_auprc = test_metrics['auprc']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'test_metrics': test_metrics,
                        'val_avg_auprc': best_score
                    }, os.path.join(self.save_dir, f'{self.model_name}.pth'))
                    cur_early_stop = 0
                elif avg_score <= best_score and val_auprc_score > best_score:
                    best_score = val_auprc_score
                    test_metrics = self.evaluate(test_loader)
                    test_auprc = test_metrics['auprc']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'test_metrics': test_metrics,
                        'val_avg_auprc': best_score
                    }, os.path.join(self.save_dir, f'{self.model_name}.pth'))
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1
                if cur_early_stop == self.n_early_stop:
                    print(f'\nEarly stopping triggered. Best val AUPRC: {best_score:.5f}, Best test AUPRC: {test_auprc:.5f}')
                    break
                print(f"[{step+1}] val_auprc: {val_auprc_score:.5f}, avg_score: {avg_score:.5f}, best_score: {best_score:.5f}, test_auprc: {test_auprc:.5f}", flush=True)
                print("----------------Training----------------", flush=True)
                self.model.train()
            if step == self.n_steps: break
        checkpoint = torch.load(os.path.join(self.save_dir, f'{self.model_name}.pth'), map_location=self.device, weights_only=False)
        return checkpoint

    def evaluate(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            all_logits, all_labels = [], [] 
            for inputs, labels, _ in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).to(torch.long)
                if isinstance(self.model, (MLP, AgeAwareMLP1, AgeAwareMLP2)):
                    logits = self.model(inputs)
                elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                    logits = self.model(inputs, self.snp_ids, self.batched_g)

                all_logits.append(logits.detach())
                all_labels.append(labels.detach())
            logits = torch.cat(all_logits).reshape(-1)
            labels = torch.cat(all_labels).reshape(-1)
        return {
            'auroc': self.auroc(logits, labels).item(),
            'auprc': self.ap(logits, labels).item(),
            'predictions': logits,
            'labels': labels
        }

class Trainer_g:
    def __init__(self, model, criterion, optimizer, device, model_name, save_dir, snp_ids, batched_g, gene_g=None, norm_weight=1.0, eval_interval=10, n_steps=20000, n_early_stop=10, log_interval=20):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.snp_ids = torch.tensor(snp_ids)
        self.batched_g = batched_g
        self.gene_g = gene_g
        self.auroc = AUROC(task='binary').to(device)
        self.ap = AveragePrecision(task='binary').to(device)
        self.eval_interval = eval_interval
        self.n_steps = n_steps
        self.n_early_stop = n_early_stop
        self.log_interval = log_interval
        self.norm_weight = norm_weight
        self.gradnorm_queue = GradientQueue(maxlen=32)
        for _ in range(self.gradnorm_queue.maxlen):
            self.gradnorm_queue.add(1.0)

    def train(self, train_loader, val_loader, test_loader, es_window=3, train_eval_loader=None):
        print(f"----------------Training {self.model_name} ----------------")
        running_loss, running_loss1, running_loss2, running_loss3, val_scores = [], [], [], [], []
        cur_early_stop, best_score, step = 0, 0, 0
        
        for inputs, labels, _ in train_loader:
            self.model.train()
            step += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            
            if isinstance(self.model, UGP_v1):
                outputs, _ = self.model(inputs, self.snp_ids, self.batched_g)
                loss = self.criterion(outputs, labels) # + self.norm_weight * torch.mean(torch.abs(weights))
                loss.backward()
                gradient_clipping(self.model, self.gradnorm_queue)
                self.optimizer.step()
                running_loss.append(loss.item())
            elif isinstance(self.model, UGP_v2):
                linear_preds, nonlinear_preds, weights = self.model(inputs, self.snp_ids, self.batched_g)
                final_preds = linear_preds + nonlinear_preds
                loss1 = self.criterion(linear_preds, labels)
                loss2 = self.criterion(nonlinear_preds, labels)
                loss3 = self.criterion(final_preds, labels)
                loss = loss1 + loss2 + loss3 #+ self.norm_weight * torch.mean(torch.abs(weights))
                loss.backward()
                gradient_clipping(self.model, self.gradnorm_queue)
                self.optimizer.step()
                running_loss.append(loss.item())
                running_loss1.append(loss1.item())
                running_loss2.append(loss2.item())
                running_loss3.append(loss3.item())
            elif isinstance(self.model, UGP_v3):
                preds, weights, attention_weights = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                loss = self.criterion(preds, labels)
                loss.backward()
                gradient_clipping(self.model, self.gradnorm_queue)
                self.optimizer.step()
                running_loss.append(loss.item())
            
            if (step + 1) % self.log_interval == 0:
                if isinstance(self.model, UGP_v1):
                    print(f"[{step + 1}] loss: {np.mean(running_loss):.3f}", flush=True)
                    running_loss = []
                elif isinstance(self.model, UGP_v3):
                    print(f"[{step + 1}] loss: {np.mean(running_loss):.3f}", flush=True)
                    running_loss = []
                else:
                    print(f"[{step + 1}] loss: {np.mean(running_loss):.3f}, loss1: {np.mean(running_loss1):.3f}, loss2: {np.mean(running_loss2):.3f}, loss3: {np.mean(running_loss3):.3f}, weight: {torch.mean(torch.abs(weights)).item()}", flush=True)
                    running_loss, running_loss1, running_loss2, running_loss3 = [], [], [], []
            
            if (step + 1) % self.eval_interval == 0:
                print("----------------Validating----------------", flush=True)
                val_metrics = self.evaluate(val_loader)
                val_auprc_score = val_metrics['auprc']
                val_scores.append(val_auprc_score)
                if len(val_scores) > es_window: 
                    val_scores.pop(0)
                avg_score = np.mean(val_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    test_metrics = self.evaluate(test_loader)
                    test_auprc = test_metrics['auprc']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'test_metrics': test_metrics,
                        'val_avg_auprc': best_score
                    }, os.path.join(self.save_dir, f'{self.model_name}.pth'))
                    cur_early_stop = 0
                elif avg_score <= best_score and val_auprc_score > best_score:
                    best_score = val_auprc_score
                    test_metrics = self.evaluate(test_loader)
                    test_auprc = test_metrics['auprc']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'test_metrics': test_metrics,
                        'val_avg_auprc': best_score
                    }, os.path.join(self.save_dir, f'{self.model_name}.pth'))
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1
                if cur_early_stop == self.n_early_stop:
                    print(f'\nEarly stopping triggered. Best val AUPRC: {best_score:.5f}, Best test AUPRC: {test_auprc:.5f}')
                    break
                print(f"[{step+1}] val_auprc: {val_auprc_score:.5f}, avg_score: {avg_score:.5f}, best_score: {best_score:.5f}, test_auprc: {test_auprc:.5f}", flush=True)
                print("----------------Training----------------", flush=True)
                self.model.train()
            if step == self.n_steps: break      
        checkpoint = torch.load(os.path.join(self.save_dir, f'{self.model_name}.pth'), map_location=self.device, weights_only=False)
        return checkpoint

    def evaluate(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            all_logits, all_labels = [], []
            for inputs, labels, _ in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).to(torch.long)
                if isinstance(self.model, UGP_v1):
                    logits, _ = self.model(inputs, self.snp_ids, self.batched_g)
                elif isinstance(self.model, UGP_v2):
                    linear_preds, nonlinear_preds, _ = self.model(inputs, self.snp_ids, self.batched_g)
                    logits = linear_preds + nonlinear_preds
                elif isinstance(self.model, UGP_v3):
                    logits, _, _ = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                all_logits.append(logits.detach())
                all_labels.append(labels.detach())
            logits = torch.cat(all_logits).reshape(-1)
            labels = torch.cat(all_labels).reshape(-1)
        return {
            'auroc': self.auroc(logits, labels).item(),
            'auprc': self.ap(logits, labels).item(),
            'predictions': logits,
            'labels': labels
        }

class KDTrainer:
    def __init__(self, model, teacher_model, criterion, optimizer, device, model_name, teacher_model_path, save_dir, 
                 snp_ids=None, batched_g=None, gene_g=None, norm_weight=1.0, eval_interval=10, n_steps=20000, n_early_stop=10, log_interval=20):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.norm_weight = norm_weight
        self.snp_ids = snp_ids
        self.batched_g = batched_g
        self.gene_g = gene_g
        self.auroc = AUROC(task='binary').to(device)
        self.ap = AveragePrecision(task='binary').to(device)
        self.eval_interval = eval_interval
        self.n_steps = n_steps
        self.n_early_stop = n_early_stop
        self.log_interval = log_interval
        self.gradnorm_queue = GradientQueue(maxlen=32)
        for _ in range(self.gradnorm_queue.maxlen):
            self.gradnorm_queue.add(1.0)
        # load best teacher model
        teacher_ckpt = torch.load(teacher_model_path, map_location=device, weights_only=False)
        teacher_model.load_state_dict(teacher_ckpt['model_state_dict'])
        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()

    def train(self, train_loader, val_loader, test_loader, es_window=3, train_eval_loader=None):
        print(f"\n{'-'*20} Training Student model with best Teacher {'-'*20}")
        running_loss, running_loss1, running_loss2, running_loss3, val_scores = [], [], [], [], []
        cur_early_stop, best_score, step = 0, 0, 0
        
        for batch in train_loader:
            self.model.train()
            step += 1
            inputs, labels, ages = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            ages = ages.to(self.device)
            self.optimizer.zero_grad()
            self.teacher_model.eval()
            # teacher
            with torch.no_grad():
                if isinstance(self.teacher_model, UGP_v1):
                    teacher_logits, _ = self.teacher_model(inputs, self.snp_ids, self.batched_g)
                elif isinstance(self.teacher_model, UGP_v2):
                    linear_preds, nonlinear_preds, _ = self.teacher_model(inputs, self.snp_ids, self.batched_g)
                    teacher_logits = linear_preds + nonlinear_preds
                elif isinstance(self.teacher_model, UGP_v3):
                    teacher_logits, _, _ = self.teacher_model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                elif isinstance(self.teacher_model, (AgeAwareMLP1, AgeAwareMLP2)):
                    teacher_logits, _, _ = self.teacher_model(inputs, ages, labels)
                elif isinstance(self.teacher_model, (AgeUGP_v1, AgeUGP_v2)):
                    teacher_logits, _, _ = self.teacher_model(inputs, self.snp_ids, self.batched_g, ages, labels)
                else: # MLP
                    teacher_logits = self.teacher_model(inputs)
            # student
            if isinstance(self.model, UGP_v1):
                student_logits, _ = self.model(inputs, self.snp_ids, self.batched_g)
                loss = self.criterion(student_logits, labels, teacher_logits) # + self.norm_weight * torch.mean(torch.abs(weights))
                loss.backward()
                gradient_clipping(self.model, self.gradnorm_queue)
                self.optimizer.step()
                running_loss.append(loss.item())
            elif isinstance(self.model, UGP_v2):
                linear_preds, nonlinear_preds, weights = self.model(inputs, self.snp_ids, self.batched_g)
                student_logits = linear_preds + nonlinear_preds
                loss1 = self.criterion(linear_preds, labels, teacher_logits)
                loss2 = self.criterion(nonlinear_preds, labels, teacher_logits)
                loss3 = self.criterion(student_logits, labels, teacher_logits)
                loss = loss1 + loss2 + loss3 #+ self.norm_weight * torch.mean(torch.abs(weights))
                loss.backward()
                gradient_clipping(self.model, self.gradnorm_queue)
                self.optimizer.step()
                running_loss.append(loss.item())
                running_loss1.append(loss1.item())
                running_loss2.append(loss2.item())
                running_loss3.append(loss3.item())
            elif isinstance(self.model, (AgeAwareMLP1, AgeAwareMLP2)):
                student_logits, _, norm_loss = self.model(inputs, ages, labels)
                loss = self.criterion(student_logits, labels, teacher_logits) + self.norm_weight * norm_loss
                loss.backward()
                gradient_clipping(self.model, self.gradnorm_queue)
                self.optimizer.step()
                running_loss.append(loss.item())
            elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                student_logits, _, norm_loss = self.model(inputs, self.snp_ids, self.batched_g, ages, labels)
                loss = self.criterion(student_logits, labels, teacher_logits) + self.norm_weight * norm_loss
                loss.backward()
                gradient_clipping(self.model, self.gradnorm_queue)
                self.optimizer.step()
                running_loss.append(loss.item())
            elif isinstance(self.model, UGP_v3):
                preds, weights, attention_weights = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                loss = self.criterion(preds, labels, teacher_logits)
                loss.backward()
                gradient_clipping(self.model, self.gradnorm_queue)
                self.optimizer.step()
                running_loss.append(loss.item())
            else: # MLP
                student_logits = self.model(inputs)
                loss = self.criterion(student_logits, labels, teacher_logits)
                loss.backward()
                gradient_clipping(self.model, self.gradnorm_queue)
                self.optimizer.step()
                running_loss.append(loss.item())

            if (step + 1) % self.log_interval == 0:
                if isinstance(self.model, UGP_v2):
                    print(f"[{step + 1}] loss: {np.mean(running_loss):.3f}, loss1: {np.mean(running_loss1):.3f}, loss2: {np.mean(running_loss2):.3f}, loss3: {np.mean(running_loss3):.3f}, weight: {torch.mean(torch.abs(weights)).item()}", flush=True)
                    running_loss, running_loss1, running_loss2, running_loss3 = [], [], [], []
                elif isinstance(self.model, UGP_v3):
                    print(f"[{step + 1}] loss: {np.mean(running_loss):.3f}", flush=True)
                    running_loss = []
                else:
                    print(f"[{step + 1}] loss: {np.mean(running_loss):.3f}", flush=True)
                    running_loss = []
            if (step + 1) % self.eval_interval == 0:
                print("----------------Validating----------------", flush=True)
                val_metrics = self.evaluate(val_loader)
                val_auprc_score = val_metrics['auprc']
                val_scores.append(val_auprc_score)
                if len(val_scores) > es_window: 
                    val_scores.pop(0)
                avg_score = np.mean(val_scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    test_metrics = self.evaluate(test_loader)
                    test_auprc = test_metrics['auprc']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'test_metrics': test_metrics,
                        'val_avg_auprc': best_score
                    }, os.path.join(self.save_dir, f'{self.model_name}.pth'))
                    cur_early_stop = 0
                elif avg_score <= best_score and val_auprc_score > best_score:
                    best_score = val_auprc_score
                    test_metrics = self.evaluate(test_loader)
                    test_auprc = test_metrics['auprc']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'test_metrics': test_metrics,
                        'val_avg_auprc': best_score
                    }, os.path.join(self.save_dir, f'{self.model_name}.pth'))
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1
                if cur_early_stop == self.n_early_stop:
                    print(f'\nEarly stopping triggered. Best val AUPRC: {best_score:.5f}, Best test AUPRC: {test_auprc:.5f}')
                    break
                    
                print(f"[{step+1}] val_auprc: {val_auprc_score:.5f}, avg_score: {avg_score:.5f}, best_score: {best_score:.5f}, test_auprc: {test_auprc:.5f}", flush=True)
                print("----------------Training----------------", flush=True)
                self.model.train()
            if step == self.n_steps: 
                break
        checkpoint = torch.load(os.path.join(self.save_dir, f'{self.model_name}.pth'), map_location=self.device, weights_only=False)
        return checkpoint

    def evaluate(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            all_logits, all_labels = [], []
            for batch in data_loader:
                inputs, labels, _ = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).to(torch.long)
                if isinstance(self.model, UGP_v1):
                    logits, _ = self.model(inputs, self.snp_ids, self.batched_g)
                elif isinstance(self.model, UGP_v2):
                    linear_preds, nonlinear_preds, _ = self.model(inputs, self.snp_ids, self.batched_g)
                    logits = linear_preds + nonlinear_preds
                elif isinstance(self.model, UGP_v3):
                    logits, _, _ = self.model(inputs, self.snp_ids, self.batched_g, self.gene_g)
                elif isinstance(self.model, (MLP, AgeAwareMLP1, AgeAwareMLP2)):
                    logits = self.model(inputs)
                elif isinstance(self.model, (AgeUGP_v1, AgeUGP_v2)):
                    logits = self.model(inputs, self.snp_ids, self.batched_g)

                all_logits.append(logits.detach())
                all_labels.append(labels.detach())
            logits = torch.cat(all_logits).reshape(-1)
            labels = torch.cat(all_labels).reshape(-1)
        return {
            'auroc': self.auroc(logits, labels).item(),
            'auprc': self.ap(logits, labels).item(),
            'predictions': logits,
            'labels': labels
        }