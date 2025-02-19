import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from src.teacher_models import AgeAwareMLP1, AgeAwareMLP2, AgeAwareMLP3

class Trainer:
    def __init__(self, model, criterion, optimizer, device, model_name, save_dir, 
                 norm_weight=1.0, eval_interval=10, n_steps=20000, n_early_stop=10, log_interval=20):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.eval_interval = eval_interval
        self.n_steps = n_steps
        self.n_early_stop = n_early_stop
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.norm_weight = norm_weight
        os.makedirs(self.save_dir, exist_ok=True)

    def evaluate(self, data_loader, desc=None):
        self.model.eval()
        all_logits = []
        all_labels = []
        
        total_batches = len(data_loader)
        
        # debug
        print(f"Evaluating with batch_size={data_loader.batch_size}, total_batches={total_batches}")

        with torch.no_grad():
            for i, (inputs, labels, _) in enumerate(data_loader):
                if desc:
                    print(f"\r{desc} {i+1}/{total_batches}", end="", flush=True)
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                
                    
                all_logits.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        if desc:
            print()

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)
        all_probs = 1 / (1 + np.exp(-all_logits))
        
        return {
            'auroc': roc_auc_score(all_labels, all_probs),
            'auprc': average_precision_score(all_labels, all_probs),
            'predictions': all_logits,
            'labels': all_labels
        }

    def train(self, train_loader, val_loader, test_loader, train_eval_loader=None):
        print(f"\n{'-'*20} Training {self.model_name} {'-'*20}")
        running_loss = []
        cur_early_stop = 0
        val_scores = []
        best_score = 0
        step = 0
        while step < self.n_steps:
            self.model.train()
            
            for inputs, labels, ages in train_loader:
                if step >= self.n_steps:
                    break
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ages = ages.to(self.device)

                self.optimizer.zero_grad()
                
                if hasattr(self, 'teacher_model'): # student
                    self.teacher_model.eval()
                    with torch.no_grad():
                        if isinstance(self.teacher_model, (AgeAwareMLP1, AgeAwareMLP2, AgeAwareMLP3)):
                            teacher_logits, _, _ = self.teacher_model(inputs, ages, labels)
                        else:
                            teacher_logits = self.teacher_model(inputs)
                            
                    if isinstance(self.model, (AgeAwareMLP1, AgeAwareMLP2, AgeAwareMLP3)):
                        student_logits, _, norm_loss = self.model(inputs, ages, labels)
                        loss = self.criterion(student_logits, labels, teacher_logits) + self.norm_weight * norm_loss
                    else:
                        student_logits = self.model(inputs)
                        loss = self.criterion(student_logits, labels, teacher_logits)
                else:  # teacher
                    if isinstance(self.model, (AgeAwareMLP1, AgeAwareMLP2, AgeAwareMLP3)):
                        outputs, _, norm_loss = self.model(inputs, ages, labels)
                        loss = self.criterion(outputs, labels) + self.norm_weight * norm_loss
                        # outputs, _= self.model(inputs, ages, labels)
                        # loss = self.criterion(outputs, labels)
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss.append(loss.item())
                
                if (step + 1) % self.log_interval == 0:
                    print(f"[{step + 1}] loss: {np.mean(running_loss):.3f}", flush=True)
                    running_loss = []
                
                if (step + 1) % self.eval_interval == 0:
                    print("----------------Validating----------------", flush=True)
                    val_metrics = self.evaluate(val_loader)
                    val_auprc_score = val_metrics['auprc']
                    
                    val_scores.append(val_auprc_score)
                    if len(val_scores) > 3: 
                        val_scores.pop(0)
                    avg_score = np.mean(val_scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        test_metrics = self.evaluate(test_loader)
                        best_test_auprc = test_metrics['auprc']
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'test_metrics': test_metrics,
                            'val_avg_auprc': best_score
                        }, os.path.join(self.save_dir, f'{self.model_name}.pth'))
                        cur_early_stop = 0

                    elif avg_score <= best_score and val_auprc_score > best_score:
                        best_score = val_auprc_score
                        test_metrics = self.evaluate(test_loader)
                        best_test_auprc = test_metrics['auprc']
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'test_metrics': test_metrics,
                            'val_avg_auprc': best_score
                        }, os.path.join(self.save_dir, f'{self.model_name}.pth'))
                        cur_early_stop = 0
                        
                    else:
                        cur_early_stop += 1

                    if cur_early_stop >= self.n_early_stop:
                        print(f'\nEarly stopping triggered. Best avg AUPRC: {best_score:.5f}, Best test AUPRC: {best_test_auprc:.5f}')
                        break
                    
                    print(f"[{step+1}] val_auprc: {val_auprc_score:.5f}, avg_score: {avg_score:.5f}, best_avg: {best_score:.5f}, best_test: {best_test_auprc:.5f}", flush=True)
                    print("----------------Training----------------", flush=True)
                    self.model.train()
                
                step += 1
                if step >= self.n_steps:
                    break
            
            if cur_early_stop == self.n_early_stop:
                break
        
        # Evaluate training set
        if train_eval_loader is not None:
            print("\nLoading best model and evaluating training set...")
            checkpoint = torch.load(os.path.join(self.save_dir, f'{self.model_name}.pth'), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            train_metrics = self.evaluate(train_eval_loader, desc="Evaluating training set:")
            checkpoint['train_metrics'] = train_metrics
            torch.save(checkpoint, os.path.join(self.save_dir, f'{self.model_name}.pth'))
            print("Evaluation complete and results saved.")
        else:
            print("No train_eval_loader provided. Skipping training set evaluation.")
            checkpoint = torch.load(os.path.join(self.save_dir, f'{self.model_name}.pth'))
        return checkpoint

class KDTrainer(Trainer):
    def __init__(self, model, teacher_model, criterion, optimizer, device, model_name, teacher_model_path, save_dir, eval_interval=10, n_steps=20000, n_early_stop=10, log_interval=20, norm_weight=1.0):
        super().__init__(model, criterion, optimizer, device, model_name, save_dir, norm_weight=norm_weight, eval_interval=eval_interval, n_steps=n_steps, n_early_stop=n_early_stop, log_interval=log_interval)
        # Load the best teacher model
        teacher_ckpt = torch.load(teacher_model_path, map_location=device)
        teacher_model.load_state_dict(teacher_ckpt['model_state_dict'])
        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()

    def train(self, train_loader, val_loader, test_loader, train_eval_loader=None):
        print(f"\n{'-'*20} Training Student Model with Best Teacher {'-'*20}")
        return super().train(train_loader, val_loader, test_loader, train_eval_loader)