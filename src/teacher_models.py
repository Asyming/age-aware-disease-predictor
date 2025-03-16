from torch import nn
import torch.nn.functional as F
import torch
from src.utils import generate_cumulative_ages

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return ctx.alpha * grad_output.neg(), None

class MLP(nn.Module):
    def __init__(self, d_input,d_hidden):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(d_hidden, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4, 1)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        preds = self.model(x)
        return preds



class AgeAwareMLP1(MLP):
    """MLP model with age-aware layer and adversarial training"""
    def __init__(self, d_input, d_hidden, use_adversarial=True, use_consist=True, use_cumulative_rate=False):
        super().__init__(d_input, d_hidden)
        self._initialize_weights()
        self.use_adversarial = use_adversarial
        self.use_consist = use_consist
        self.use_cumulative_rate = use_cumulative_rate
        self.age_layer = nn.Sequential(
            nn.Linear(1, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4, 2)
        )
        
        self.age_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        with torch.no_grad():
            nn.init.normal_(self.age_layer[-1].weight, mean=0, std=0.01)
            self.age_layer[-1].bias.data.copy_(torch.tensor([0., 1.]))
    
    def _initialize_weights(self):
        skip_layers = []
        if hasattr(self, 'age_layer'):
            skip_layers.append(self.age_layer[-1])

        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in skip_layers:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_transition_matrix(self, age_norm, age_cumulative_norm=None):
        if age_norm.dim() == 1:
            age_norm = age_norm.unsqueeze(1)

        if age_cumulative_norm is not None:
            pos_trans = self.age_layer(age_cumulative_norm)
        else:
            pos_trans = self.age_layer(age_norm)

        age_ori = age_norm * (70 - 40) + 40

        pos_probs = F.softmax(pos_trans, dim=1)
        # debug
        if torch.rand(1).item() < 0.01:
            print(f"Age: {age_ori[0].item():.1f}")
            print(f"Before softmax: {pos_trans[0]}")
            print(f"After softmax: {pos_probs[0]}")
        batch_size = age_norm.shape[0]
        trans_matrix = torch.zeros(batch_size, 2, 2, device=age_norm.device)
        trans_matrix[:, 0, 0] = 1.0    # p00 = 1.0
        trans_matrix[:, 0, 1] = 0.0    # p01 = 0.0
        trans_matrix[:, 1, 0] = pos_probs[:, 0]  #.clamp(0, 0.5) # p10
        trans_matrix[:, 1, 1] = pos_probs[:, 1]  # p11
        return trans_matrix

    def forward(self, x, age=None, labels=None):
        original_logits = super().forward(x)
        assert not torch.isnan(original_logits).any(), "[AgeAwareMLP1] lr maybe too high!"
        
        if age is None or labels is None:
            return original_logits
        
        if self.use_cumulative_rate:
            age_cumulative = generate_cumulative_ages(age, mask=labels)
            age_cumulative_norm = (age_cumulative - 40) / (70 - 40)

        features = None
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == 6:
                features = x

        age_norm = (age - 40) / (70 - 40)
        consist_weight = 0.0
        age_weight = 0.0

        if self.use_adversarial:
            features_rev = GradientReverseLayer.apply(features, 0.001)
            age_pred = self.age_predictor(features_rev)
            age_weight = 0.5
            age_loss = F.l1_loss(age_pred.squeeze(), age_norm) # always use original age
        else:
            age_loss = 0.0
        
        trans_matrix = self.get_transition_matrix(age_norm, age_cumulative_norm) if self.use_cumulative_rate else self.get_transition_matrix(age_norm) # use cumulative age
        original_probs = torch.sigmoid(original_logits)
        original_probs = torch.clamp(original_probs, 1e-7, 1-1e-7) # avoid log(0)
        neg_mask = (1 - labels.float())
        
        neg_dist = torch.cat([1 - original_probs, original_probs], dim=1)
        updated_dist = torch.bmm(neg_dist.unsqueeze(1), trans_matrix)
        updated_probs = updated_dist.squeeze(1)[:, 1:2]
        
        assert len(updated_probs[labels==1]) > 0
        consistency_loss = F.binary_cross_entropy(updated_probs[labels==1], original_probs[labels==1])

        final_probs = (1 - neg_mask) * original_probs + neg_mask * updated_probs
        final_logits = torch.log(final_probs / (1 - final_probs + 1e-7))


        if self.use_consist:
            consist_weight = 1.0
            
        final_loss = consist_weight * consistency_loss + age_weight * age_loss
        return final_logits, original_logits, final_loss

class AgeAwareMLP2(MLP):
    """MLP model with feature disentanglement for age-related features"""
    def __init__(self, d_input, d_hidden, use_ageloss=True, use_disentangle=True, use_consist=True, use_cumulative_rate=False):
        super().__init__(d_input, d_hidden)
        self._initialize_weights()
        self.use_ageloss = use_ageloss
        self.use_disentangle = use_disentangle
        self.use_consist = use_consist
        self.use_cumulative_rate = use_cumulative_rate
        self.feature_dim = 16
        self.main_dim = 15
        self.age_dim = 1
        
        self.age_predictor = nn.Sequential(
            nn.Linear(self.age_dim, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        self.main_head = nn.Sequential(
            nn.Linear(self.main_dim, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4, 1)
        )
        
        self.age_layer = nn.Sequential(
            nn.Linear(2, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4, 2)
        )

        self._initialize_weights()
        with torch.no_grad():
            nn.init.normal_(self.age_layer[-1].weight, mean=0, std=0.01)
            self.age_layer[-1].bias.data.copy_(torch.tensor([0., 1.]))

    def _initialize_weights(self):
        skip_layers = []
        if hasattr(self, 'age_layer'):
            skip_layers.append(self.age_layer[-1])

        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in skip_layers:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def get_intermediate_features(self, x):
        x = self.model[:7](x)
        main_feat = x[:, :self.main_dim]
        age_feat = x[:, self.main_dim:]
        return main_feat, age_feat

    def get_transition_matrix(self, age_norm, age_feat, age_cumulative_norm=None):
        if age_norm.dim() == 1:
            age_norm = age_norm.unsqueeze(1)

        if age_cumulative_norm is not None:
            combined_input = torch.cat([age_cumulative_norm, age_feat], dim=1)
        else:
            combined_input = torch.cat([age_norm, age_feat], dim=1)

        age_ori = age_norm * (70 - 40) + 40

        pos_trans = self.age_layer(combined_input)
        pos_probs = F.softmax(pos_trans, dim=1)

        if torch.rand(1).item() < 0.01:
            print(f"Age: {age_ori[0].item():.1f}")
            print(f"Before softmax: {pos_trans[0]}")
            print(f"After softmax: {pos_probs[0]}")

        batch_size = age_norm.shape[0]
        trans_matrix = torch.zeros(batch_size, 2, 2, device=age_norm.device)
        trans_matrix[:, 0, 0] = 1.0
        trans_matrix[:, 0, 1] = 0.0
        trans_matrix[:, 1, 0] = pos_probs[:, 0] # p10
        trans_matrix[:, 1, 1] = pos_probs[:, 1] # p11
        
        return trans_matrix

    def forward(self, x, age=None, labels=None):
        main_feat, age_feat = self.get_intermediate_features(x)
        
        original_logits = self.main_head(main_feat)
        assert not torch.isnan(original_logits).any(), "[AgeAwareMLP2] lr maybe too high!"
        
        if age is None or labels is None:
            return original_logits
        
        if self.use_cumulative_rate:
            age_cumulative = generate_cumulative_ages(age, mask=labels)
            age_cumulative_norm = (age_cumulative - 40) / (70 - 40)

        age_norm = (age - 40) / (70 - 40)
        age_pred = self.age_predictor(age_feat)
        age_loss = F.l1_loss(age_pred.squeeze(), age_norm) # always use original age
        
        batch_size = main_feat.size(0)
        main_feat_normalized = F.normalize(main_feat, dim=1)
        correlation = torch.mm(main_feat_normalized.t(), age_feat)
        disentangle_loss = torch.norm(correlation, p='fro') / (batch_size * (main_feat.size(1) + age_feat.size(1)))

        trans_matrix = self.get_transition_matrix(age_norm, age_feat, age_cumulative_norm) if self.use_cumulative_rate else self.get_transition_matrix(age_norm, age_feat) # use cumulative age
        original_probs = torch.sigmoid(original_logits)
        original_probs = torch.clamp(original_probs, 1e-7, 1-1e-7) # avoid log(0)
        neg_mask = (1 - labels.float())
        
        neg_dist = torch.cat([1 - original_probs, original_probs], dim=1)
        updated_dist = torch.bmm(neg_dist.unsqueeze(1), trans_matrix)
        updated_probs = updated_dist.squeeze(1)[:, 1:2]
        assert len(updated_probs[labels==1]) > 0
        consistency_loss = F.binary_cross_entropy(updated_probs[labels==1], original_probs[labels==1])

        final_probs = (1 - neg_mask) * original_probs + neg_mask * updated_probs
        final_logits = torch.log(final_probs / (1 - final_probs + 1e-7))
        
        consist_weight = 0.0
        age_weight = 0.0
        disentangle_weight = 0.0

        if self.use_consist:
            consist_weight = 1.0
        if self.use_ageloss:
            age_weight = 0.5
        if self.use_disentangle:
            disentangle_weight = 0.5

        final_loss = consist_weight * consistency_loss + age_weight * age_loss + disentangle_weight * disentangle_loss

        return final_logits, original_logits, final_loss