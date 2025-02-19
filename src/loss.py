import torch
from torch import nn
import torch.nn.functional as F

# class KDLoss(nn.Module):
#     def __init__(self, temperature=2.0, alpha=0.1):
#         super().__init__()
#         self.T = temperature
#         self.alpha = alpha
#         self.bce = nn.BCEWithLogitsLoss()
        
#     def forward(self, student_logits, labels, teacher_outputs):
#         teacher_logits = teacher_outputs  
#         # Hard Loss
#         hard_loss = self.bce(student_logits, labels)
#         # Soft Loss
#         soft_student = student_logits / self.T
#         soft_teacher = teacher_logits.detach() / self.T
#         soft_loss = F.binary_cross_entropy_with_logits(
#             soft_student,
#             torch.sigmoid(soft_teacher)
#         )
#         # Total Loss
#         loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss * (self.T ** 2)
        
#         return loss
    
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

        soft_loss_scaled = soft_loss * (self.T ** 2)
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss_scaled

        return total_loss
