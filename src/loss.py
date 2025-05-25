import torch
from torch import nn

class KDLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.1):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, labels, teacher_logits):
        hard_loss = self.bce(student_logits, labels)

        teacher_logits = teacher_logits.detach()
        scaled_teacher = teacher_logits / self.T
        scaled_student = student_logits / self.T
        teacher_probs = torch.sigmoid(scaled_teacher).clamp(1e-7, 1-1e-7)
        student_probs = torch.sigmoid(scaled_student).clamp(1e-7, 1-1e-7)
        kl_positive = teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs))
        kl_negative = (1 - teacher_probs) * (torch.log(1 - teacher_probs) - torch.log(1 - student_probs))
        soft_loss = (kl_positive + kl_negative).mean()

        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss * (self.T ** 2)
        return total_loss
    
class BGCELoss(nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, labels):
        p = torch.sigmoid(logits)
        p_j = labels * p + (1 - labels) * (1 - p)
        if self.q == 0:
            loss = self.bce(logits, labels)
        else:
            loss = ((1 - torch.pow(p_j, self.q)) / self.q).mean()
        return loss
    
# TODO: add peer loss/DMI loss/SCE loss
class PeerLoss(nn.Module):
    def __init__(self):
        super().__init__()

class DMILoss(nn.Module):
    def __init__(self):
        super().__init__()

class SCELoss(nn.Module):
    def __init__(self):
        super().__init__()