from torch import nn
import torch.nn.functional as F
import torch
import dgl

def bert_init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm1d):
        if module.weight is not None:
            torch.nn.init.normal_(module.weight.data, mean=1, std=0.02)
        if module.weight is not None:
            torch.nn.init.constant_(module.bias.data, 0)

def generate_cumulative_ages(original_tensor, mask, max_age=70, min_age=40):
    mask = mask.bool()
    device = original_tensor.device
    mask = mask.to(device)

    upper_bounds = torch.where(mask, max_age, original_tensor)
    lower_bounds = torch.where(mask, original_tensor, min_age)
    ranges = upper_bounds - lower_bounds + 1

    rand = torch.rand(original_tensor.shape, device=device)
    random_offsets = (rand * ranges.float()).long()
    cumulative_ages = lower_bounds + random_offsets

    return cumulative_ages

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

            nn.Linear(16, 1)
        )
        self.apply(lambda module: bert_init_params(module))
    def forward(self, x):
        preds = self.model(x)
        return preds

class ctrMLP(nn.Module):
    def __init__(self, d_input, d_hidden=64, ema=0.99, pos_tau=0.99, neg_tau=0.70, neg_rate=1.0, lamb=1.0):
        super().__init__()
        self.ema = ema
        self.pos_tau = pos_tau
        self.neg_tau = neg_tau
        self.neg_rate = neg_rate
        self.lamb = lamb
        self.backbone = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.projector = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.BatchNorm1d(d_hidden * 2),
            nn.ReLU(),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Dropout(0.5)
        )
        
        self.backbone_k = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.projector_k = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.BatchNorm1d(d_hidden * 2),
            nn.ReLU(),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            self.backbone,
            nn.Linear(d_hidden, 1)
        )

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.encoder_k = nn.Sequential(
            self.backbone_k,
            self.projector_k
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.BatchNorm1d(d_hidden * 2),
            nn.ReLU(),
            nn.Linear(d_hidden * 2, d_hidden)
        )
        
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        self.apply(lambda module: bert_init_params(module))
        
    def forward(self, x, teacher_logits=None, ages=None, labels=None):
        ori_logits = self.classifier(x) # x3
        if labels is None:
            return ori_logits, 0

        batch_size = x.size(0)
        #x1, x2 = x
        z1 = self.encoder(x) # x1
        p1 = self.predictor(z1)
        p1 = F.normalize(p1, dim=1)
        
        with torch.no_grad():
            m = self.ema
            for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)
        z2 = self.encoder_k(x) # x2
        z2 = F.normalize(z2, dim=1)
        
        p1 = torch.clamp(p1, 1e-4, 1.0 - 1e-4)
        z2 = torch.clamp(z2, 1e-4, 1.0 - 1e-4)
        
        contrast_sim = torch.matmul(p1, z2.t())  # [batch_size, batch_size]
        # -<q,z> + log(1-<q,z>)
        contrast_loss = -contrast_sim * torch.eye(batch_size).cuda() + torch.log(1 - contrast_sim) * (1 - torch.eye(batch_size).cuda())
        if teacher_logits is not None:
            probs = torch.sigmoid(teacher_logits).view(-1) # soft labels [batch_size, d_hidden]
        else:
            probs = torch.sigmoid(ori_logits).view(-1) # soft labels [batch_size, d_hidden]
        # weights: |p - 0.5| + |q - 0.5|, 0.5附近更无效, 而且越接近0.5越可能是噪声(另一种判断噪声的方法:label和prob差异大)
        #cert_matrix = torch.abs(probs_flat - 0.5).unsqueeze(1) + torch.abs(probs_flat - 0.5).unsqueeze(0)
        cert_matrix = 4 * torch.abs(probs - 0.5).unsqueeze(1) * torch.abs(probs - 0.5).unsqueeze(0)
        uncert_matrix = 1.0 - cert_matrix

        diff_matrix = torch.abs(probs.unsqueeze(1) - probs.unsqueeze(0))
        sim_matrix = 1.0 - diff_matrix

        labels_flat = labels.float().view(-1)
        label_matrix = labels_flat.unsqueeze(1) * 2 + labels_flat.unsqueeze(0) # 0: 0-0, 1: 0-1, 2: 1-0, 3: 1-1

        age_mask = (ages >= 65).float().view(-1)###
        age_matrix = age_mask.unsqueeze(1) * 2 + age_mask.unsqueeze(0) # 0: low-low, 1: low-high, 2: high-low, 3: high-high

        untrusted_label_0_mask = (label_matrix == 0.0) & (age_matrix != 3.0)
        untrusted_label_1_mask = (label_matrix == 1.0) & ((age_matrix == 0.0) | (age_matrix == 1.0))
        untrusted_label_2_mask = (label_matrix == 2.0) & ((age_matrix == 0.0) | (age_matrix == 2.0))
        untrusted_label_mask = untrusted_label_0_mask | untrusted_label_1_mask | untrusted_label_2_mask
        # AD: pos_tau = 0.99, neg_tau = 0.80
        # MS: pos_tau = 0.99, neg_tau = 0.70
        # UC: pos_tau = 0.99, neg_tau = 0.70
        # AF: pos_tau = 1.00, neg_tau = 0.70
        pos_mask = ((sim_matrix >= self.pos_tau) & (untrusted_label_1_mask | untrusted_label_2_mask)).float()
        neg_mask = ((sim_matrix < self.neg_tau) & untrusted_label_0_mask).float()
        mask = pos_mask - self.neg_rate * neg_mask
        # debug
        if torch.rand(1).item() < 0.01:
            print(f"pos_sum: {pos_mask.sum()}")
            print(f"neg_sum: {neg_mask.sum()}")

        sim_matrix = sim_matrix.float() / sim_matrix.abs().sum(1, keepdim=True)
        sim_matrix = (sim_matrix) * mask
        ctrr_loss = self.lamb * (contrast_loss * sim_matrix).sum(dim=1).mean(0)
        return ori_logits, ctrr_loss

class AgeAwareMLP1(nn.Module):
    """MLP model with age-aware layer and adversarial training"""
    def __init__(self, d_input, d_hidden, use_adversarial=True, use_consist=True, use_cumulative_rate=False):
        super().__init__()
        self.use_adversarial = use_adversarial
        self.use_consist = use_consist
        self.use_cumulative_rate = use_cumulative_rate
        self.feature_dim = 16
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(d_hidden, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(self.feature_dim, 1)
        
        self.age_layer = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        self.age_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )

        self.apply(lambda module: bert_init_params(module))
        with torch.no_grad():
            nn.init.normal_(self.age_layer[-1].weight, mean=0, std=0.01)
            self.age_layer[-1].bias.data.copy_(torch.tensor([0., 1.]))

    def get_transition_matrix(self, age_norm, age_cumulative_norm=None):
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
        features = self.feature_extractor(x)
        original_logits = self.predictor(features)
        assert not torch.isnan(original_logits).any(), "[AgeAwareMLP1] lr maybe too high!"
        if age is None or labels is None:
            return original_logits

        if self.use_cumulative_rate:
            age_cumulative = generate_cumulative_ages(age, mask=labels)
            age_cumulative_norm = (age_cumulative - 40) / (70 - 40)

        age_norm = (age - 40) / (70 - 40)
        neg_mask = (1 - labels.float())
        consist_weight = 0.0 if not self.use_consist else 1.0
        age_weight = 0.0
        if self.use_adversarial:
            features_rev = GradientReverseLayer.apply(features, 0.01)
            age_pred = self.age_predictor(features_rev)
            age_weight = 0.5
            age_loss = F.l1_loss(age_pred.squeeze(), age_norm) # always use original age
        else: age_loss = 0.0

        trans_matrix = self.get_transition_matrix(age_norm, age_cumulative_norm) if self.use_cumulative_rate else self.get_transition_matrix(age_norm) # use cumulative age
        original_probs = torch.sigmoid(original_logits)
        original_probs = torch.clamp(original_probs, 1e-7, 1-1e-7) # avoid log(0)
        neg_dist = torch.cat([1 - original_probs, original_probs], dim=1)
        updated_dist = torch.bmm(neg_dist.unsqueeze(1), trans_matrix)
        updated_probs = updated_dist.squeeze(1)[:, 1:2]
        assert len(updated_probs[labels==1]) > 0
        consistency_loss = F.binary_cross_entropy(updated_probs[labels==1], original_probs[labels==1])

        final_probs = (1 - neg_mask) * original_probs + neg_mask * updated_probs
        final_logits = torch.log(final_probs / (1 - final_probs + 1e-7))
        final_loss = consist_weight * consistency_loss + age_weight * age_loss
        return final_logits, original_logits, final_loss

class AgeAwareMLP2(nn.Module):
    """MLP model with feature disentanglement for age-related features"""
    def __init__(self, d_input, d_hidden, use_ageloss=True, use_disentangle=True, use_consist=True, use_cumulative_rate=False):
        super().__init__()
        self.use_ageloss = use_ageloss
        self.use_disentangle = use_disentangle
        self.use_consist = use_consist
        self.use_cumulative_rate = use_cumulative_rate
        self.feature_dim = 16
        self.main_dim = 15
        self.age_dim = 1
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(d_hidden, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
        )
        
        self.age_predictor = nn.Sequential(
            nn.Linear(self.age_dim, 1),
            nn.Sigmoid()
        )
        
        self.main_head = nn.Sequential(
            nn.Linear(self.main_dim, 1)
        )
        
        self.age_layer = nn.Sequential(
            nn.Linear(self.age_dim + 1, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        self.apply(lambda module: bert_init_params(module))
        with torch.no_grad():
            nn.init.normal_(self.age_layer[-1].weight, mean=0, std=0.01)
            self.age_layer[-1].bias.data.copy_(torch.tensor([0., 1.]))

    def get_intermediate_features(self, x):
        x = self.feature_extractor(x)
        main_feat = x[:, :self.main_dim]
        age_feat = x[:, self.main_dim:]
        return main_feat, age_feat

    def get_transition_matrix(self, age_norm, age_feat, age_cumulative_norm=None):
        if age_cumulative_norm is not None:
            combined_input = torch.cat([age_cumulative_norm, age_feat], dim=1)
        else:
            combined_input = torch.cat([age_norm, age_feat], dim=1)

        age_ori = age_norm * (70 - 40) + 40
        pos_trans = self.age_layer(combined_input)
        pos_probs = F.softmax(pos_trans, dim=1)
        # debug
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
        
        consist_weight = 0.0 if not self.use_consist else 1.0
        age_weight = 0.0 if not self.use_ageloss else 0.5
        disentangle_weight = 0.0 if not self.use_disentangle else 0.5
        final_loss = consist_weight * consistency_loss + age_weight * age_loss + disentangle_weight * disentangle_loss

        return final_logits, original_logits, final_loss

# mean filters as feat
class UGP_v1(nn.Module):
    def __init__(self, n_snps, n_genes, d_emb=8, n_filters=8, gene_dropout=0, snp_dropout=0):
        super().__init__()
        self.n_filters = n_filters
        
        self.snp_emb = nn.Embedding(10, d_emb)
        self.filter_list = nn.ParameterList()
        for _ in range(n_filters):
            self.filter_list.append(nn.Parameter(torch.randn(n_snps)*0.001)) # trainable weights: n_filter x N_genotype
        
        self.predictor = nn.Sequential(
            nn.Dropout(gene_dropout),
            nn.Linear(n_genes, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 1),
        )
        self.apply(lambda module: bert_init_params(module))
        self.dropout = nn.Dropout(snp_dropout)
    def forward(self, snp, snp_ids, g): # SNP: batch_size x N_genotype, snp_ids:N_genes[gene1_snps_ids, gene2_snps_ids, gene3_snps_ids, ...] , G: snp-gene graph
        batch_size = len(snp)
        sample_h = []
        snp = snp.reshape(batch_size,-1,1) # batch_size x N_genotype x 1
        snp_h_list = []
        for i in range(self.n_filters):
            snp_h_list.append(torch.einsum('bnd,n->bnd', self.dropout(snp), self.filter_list[i]))
        snp_h = torch.concatenate(snp_h_list, dim=-1) # all snps info: batch_size x N_genotype x n_filter  
        for i in range(batch_size): 
            # g: n_genes graph
            g.ndata['h'] = torch.index_select(snp_h[i],0,snp_ids) # N_genotype x n_filter snp_ids:N_genes[gene1_snps_ids, gene2_snps_ids, gene3_snps_ids, ...] -> matrix: gene1_snps_ids, gene2_snps_ids,  gene3_snps_ids, ...
            # SNP-> Gene: snp-gene graph, feat N_genotype plus x n_filter, dgl.readout_nodes(g, 'h', op='sum')-> n_genes x n_filter -> n_genes
            sample_h.append(torch.mean(dgl.readout_nodes(g, 'h', op='sum'),dim=-1).reshape(-1)) # readout_nodes: 要求每个node的特征数相同, 在子图上进行sum pooling, 返回n_genes x n_filter
        sample_h = torch.stack(sample_h, dim=0) # batch_size x n_genes
        preds = self.predictor(sample_h)
        filter_list = []
        for i in range(self.n_filters):
            filter_list.append(self.filter_list[i])
        return preds, torch.stack(filter_list,dim=0)

# keep filters as feat
class UGP_v2(nn.Module):
    def __init__(self, n_snps, n_genes, d_emb=8, n_filters=8, gene_dropout=0, snp_dropout=0):
        super().__init__()
        self.n_filters = n_filters
        
        self.snp_emb = nn.Embedding(10, d_emb)
        self.filter_list = nn.ParameterList()
        for _ in range(n_filters):
            self.filter_list.append(nn.Parameter(torch.randn(n_snps)*0.001))
        
        self.predictor = nn.Sequential(
            nn.Dropout(gene_dropout),
            nn.Linear(n_genes*n_filters, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 1),
        )
        self.apply(lambda module: bert_init_params(module))
        self.dropout = nn.Dropout(snp_dropout)
    def forward(self, snp, snp_ids, g):
        batch_size = len(snp) # batch_size * n_genotype
        sample_h = []
        snp = snp.reshape(batch_size,-1,1) # batch_size * n_genotype * 1
        snp_h_list = []
        for i in range(self.n_filters):
            snp_h_list.append(torch.einsum('bnd,n->bnd', self.dropout(snp), self.filter_list[i])) # ls: [(filter1)batch_size * n_genotype * 1, (filter2)batch_size * n_genotype * 1, ...]
        snp_h = torch.concatenate(snp_h_list, dim=-1) # batch_size * n_genotype * n_filters
        for i in range(batch_size):
            g.ndata['h'] = torch.index_select(snp_h[i],0,snp_ids)
            sample_h.append(dgl.readout_nodes(g, 'h', op='sum').reshape(-1))
        sample_h = torch.stack(sample_h, dim=0)
        nonlinear_preds = self.predictor(sample_h)
        filter_list = []
        for i in range(self.n_filters):
            filter_list.append(self.filter_list[i])
        linear_preds = torch.sum(torch.einsum('bnd,n->bnd', self.dropout(snp), torch.sum(torch.stack(filter_list,dim=0),dim=0)),dim=1)
        # print(linear_preds.shape, flush=True)
        return linear_preds.reshape(-1,1), nonlinear_preds.reshape(-1,1), torch.stack(filter_list,dim=0)
    
# PRS-Net GIN method
class UGP_v3(nn.Module):
    def __init__(self, n_snps, n_genes, d_hidden=64, n_filters=8, gene_dropout=0, snp_dropout=0, n_gnn_layers=2):
        super().__init__()
        self.snp_emb = nn.Embedding(10, 8)
        self.n_filters = n_filters
        self.filter_list = nn.ParameterList()
        for _ in range(n_filters):
            self.filter_list.append(nn.Parameter(torch.randn(n_snps)*0.001))
        self.gnn_layer_list = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        
        for _ in range(n_gnn_layers):
            gin_mlp = nn.Sequential(
                nn.Linear(d_hidden, d_hidden*2),
                nn.BatchNorm1d(d_hidden*2),
                nn.ReLU(),
                nn.Linear(d_hidden*2, d_hidden)
            )
            self.gnn_layer_list.append(
                dgl.nn.GINConv(gin_mlp, learn_eps=False, aggregator_type='sum')
            )
            self.batch_norm_list.append(nn.BatchNorm1d(d_hidden))
        
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_filters, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )

        self.attention_key = nn.Linear(d_hidden, d_hidden)
        self.attention_query = nn.Sequential(
            nn.Linear(d_hidden, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_value = nn.Linear(d_hidden, d_hidden)
        
        self.predictor = nn.Sequential(
            nn.Dropout(gene_dropout),
            nn.Linear(d_hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
        self.apply(lambda module: bert_init_params(module))
        self.dropout = nn.Dropout(snp_dropout)
    
    def attentive_readout(self, g, feat):
        with g.local_scope():
            keys = self.attention_key(feat)
            g.ndata['w'] = self.attention_query(keys) # QK
            g.ndata['v'] = self.attention_value(feat) # V
            h = dgl.readout.sum_nodes(g, 'v', 'w')  # [batch_size, n_filters]
            return h, g.ndata['w']
        
    def forward(self, snp, snp_ids, batched_g, gene_g): # batched_g: batch_size * n_genes * n_snps, gene_g: n_genes * n_snps
        batch_size = len(snp)
        snp = snp.reshape(batch_size, -1, 1)
        snp_h_list = []
        for i in range(self.n_filters):
            snp_h_list.append(torch.einsum('bnd,n->bnd', self.dropout(snp), self.filter_list[i])) 
        snp_h = torch.concatenate(snp_h_list, dim=-1) # batch_size * n_snps * n_filters

        gene_features_list = []
        for i in range(batch_size):
            batched_g.ndata['h'] = torch.index_select(snp_h[i], 0, snp_ids)
            gene_features = dgl.readout_nodes(batched_g, 'h', op='sum') # n_genes x n_filters
            gene_features_list.append(gene_features)
        h = torch.cat(gene_features_list, dim=0) # [batch_size * n_genes, n_filters]
        # gene encoder
        h = self.gene_encoder(h)
        batched_gene_g = dgl.batch([gene_g] * batch_size)
        # gnn
        hidden_rep = [h]
        for i in range(len(self.gnn_layer_list)):
            h = self.gnn_layer_list[i](batched_gene_g, h)
            h = self.batch_norm_list[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        # print(hidden_rep)
        # attentive readout
        g_h, weights = self.attentive_readout(batched_gene_g, hidden_rep[-1])
        weights = weights.view(batch_size, -1) # Reshape to [batch_size, n_genes]
        # prediction
        preds = self.predictor(g_h)
        filter_list = [self.filter_list[i] for i in range(self.n_filters)]
        return preds, torch.stack(filter_list, dim=0), weights

class ctrUGP_v1(nn.Module):
    def __init__(self, n_snps, n_genes, age_threshold, d_hidden=64, n_filters=8, gene_dropout=0, snp_dropout=0, n_gnn_layers=2, ema=0.99, pos_tau=0.99, neg_tau=0.7, neg_rate=1.0, lamb=1.0):
        super().__init__()
        ## UGP_v3 components
        self.age_threshold = age_threshold
        self.dropout = nn.Dropout(snp_dropout)
        self.n_filters = n_filters
        self.filter_list = nn.ParameterList()
        for _ in range(n_filters):
            self.filter_list.append(nn.Parameter(torch.randn(n_snps)*0.001))
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_filters, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        ## GNN components
        self.gnn_layer_list = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        for _ in range(n_gnn_layers):
            gin_mlp = nn.Sequential(
                nn.Linear(d_hidden, d_hidden*2),
                nn.BatchNorm1d(d_hidden*2),
                nn.ReLU(),
                nn.Linear(d_hidden*2, d_hidden)
            )
            self.gnn_layer_list.append(
                dgl.nn.GINConv(gin_mlp, learn_eps=False, aggregator_type='sum')
            )
            self.batch_norm_list.append(nn.BatchNorm1d(d_hidden))
        ## attention components
        self.attention_key = nn.Linear(d_hidden, d_hidden)
        self.attention_query = nn.Sequential(
            nn.Linear(d_hidden, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_value = nn.Linear(d_hidden, d_hidden)
        
        ###
        self.predictor_cls = nn.Sequential(
            nn.Dropout(gene_dropout),
            nn.Linear(d_hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        ## contrastive learning components
        ###
        self.ema = ema # moving average of probbility outputs
        self.pos_tau = pos_tau # contrastive threshold.
        self.neg_tau = neg_tau # contrastive threshold.
        self.neg_rate = neg_rate # negative rate
        self.lamb = lamb # weight for contrastive regularization(CTRR) term
        # encoder_q
        ###
        self.projector = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.BatchNorm1d(d_hidden * 2),
            nn.ReLU(),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Dropout(0.5)
        )
        # encoder_k (momentum encoder)
        ###
        self.projector_k = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.BatchNorm1d(d_hidden * 2),
            nn.ReLU(),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Dropout(0.5)
        )
        # contrastive predictor
        ###
        self.predictor_ctr = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.BatchNorm1d(d_hidden * 2),
            nn.ReLU(),
            nn.Linear(d_hidden * 2, d_hidden)
        )
        
        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.apply(lambda module: bert_init_params(module))
    
    def _update_momentum_encoder(self):
        with torch.no_grad():
            m = self.ema
            for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)
    
    def attentive_readout(self, g, feat):
        with g.local_scope():
            keys = self.attention_key(feat)
            g.ndata['w'] = self.attention_query(keys)
            g.ndata['v'] = self.attention_value(feat)
            h = dgl.readout.sum_nodes(g, 'v', 'w')
            return h, g.ndata['w']
        
    def forward(self, snp, snp_ids, batched_g, gene_g, teacher_logits=None, ages=None, labels=None):
        batch_size = len(snp)
        batched_gene_g = dgl.batch([gene_g] * batch_size)
        snp = snp.reshape(batch_size, -1, 1)

        snp_h_list = []
        for i in range(self.n_filters):
            snp_h_list.append(torch.einsum('bnd,n->bnd', self.dropout(snp), self.filter_list[i]))
        filter_list = [self.filter_list[i] for i in range(self.n_filters)]
        snp_h = torch.concatenate(snp_h_list, dim=-1)

        gene_features_list = []
        for i in range(batch_size):
            batched_g.ndata['h'] = torch.index_select(snp_h[i], 0, snp_ids)
            gene_features = dgl.readout_nodes(batched_g, 'h', op='sum')
            gene_features_list.append(gene_features)
        h = torch.cat(gene_features_list, dim=0)

        h = self.gene_encoder(h)
        ## GNN
        hidden_rep = [h]
        for i in range(len(self.gnn_layer_list)):
            h = self.gnn_layer_list[i](batched_gene_g, h)
            h = self.batch_norm_list[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        ## attentive readout
        g_h, weights = self.attentive_readout(batched_gene_g, hidden_rep[-1])
        weights = weights.view(batch_size, -1) # Reshape to [batch_size, n_genes]

        preds = self.predictor_cls(g_h)
        if labels is None:
            return preds, 0, 0, 0

        ## contrastive learning
        z1 = self.projector(g_h)
        p1 = self.predictor_ctr(z1)
        p1 = F.normalize(p1, dim=1)
        
        self._update_momentum_encoder()
        with torch.no_grad():
            z2 = self.projector_k(g_h)
            z2 = F.normalize(z2, dim=1)
     
        p1 = torch.clamp(p1, 1e-4, 1.0 - 1e-4)
        z2 = torch.clamp(z2, 1e-4, 1.0 - 1e-4)
        
        contrast_sim = torch.matmul(p1, z2.t())  # [batch_size, batch_size]
        contrast_loss = -contrast_sim * torch.eye(batch_size).cuda() + torch.log(1 - contrast_sim) * (1 - torch.eye(batch_size).cuda())
        
        if teacher_logits is not None:
            probs = torch.sigmoid(teacher_logits).view(-1)
        else:
            probs = torch.sigmoid(preds).view(-1)
        ## 4 types of mask matrix
        cert_matrix = 4 * torch.abs(probs - 0.5).unsqueeze(1) * torch.abs(probs - 0.5).unsqueeze(0)
        uncert_matrix = 1.0 - cert_matrix

        diff_matrix = torch.abs(probs.unsqueeze(1) - probs.unsqueeze(0))# * uncert_matrix
        sim_matrix = 1.0 - diff_matrix
        # sim_matrix = torch.sigmoid(torch.matmul(preds, preds.t())).clone().detach()
        # sim_matrix.fill_diagonal_(1)
        #print(sim_matrix)

        labels_flat = labels.float().view(-1)
        label_matrix = labels_flat.unsqueeze(1) * 2 + labels_flat.unsqueeze(0) # 0: 0-0, 1: 0-1, 2: 1-0, 3: 1-1
        ###
        age_mask = (ages >= self.age_threshold).float().view(-1)
        age_matrix = age_mask.unsqueeze(1) * 2 + age_mask.unsqueeze(0) # 0: low-low, 1: low-high, 2: high-low, 3: high-high
        ##
        # pos_mask = ((uncert_matrix <= 0.95) & (sim_matrix >= 0.95) & ((label_matrix == 1.0) | (label_matrix == 2.0))).float()
        # neg_mask = ((uncert_matrix <= 0.95) & (sim_matrix <= 0.6) & ((label_matrix == 0.0) | (label_matrix == 0.0))).float()
        untrusted_label_0_mask = (label_matrix == 0.0) & (age_matrix != 3.0)
        untrusted_label_1_mask = (label_matrix == 1.0) & ((age_matrix == 0.0) | (age_matrix == 1.0))
        untrusted_label_2_mask = (label_matrix == 2.0) & ((age_matrix == 0.0) | (age_matrix == 2.0))
        untrusted_label_mask = untrusted_label_0_mask | untrusted_label_1_mask | untrusted_label_2_mask
        # AD: pos_tau = 0.99, neg_tau = 0.70
        # MS: pos_tau = 0.99, neg_tau = 0.70
        # UC: pos_tau = 0.99, neg_tau = 0.70
        # AF: pos_tau = 1.00, neg_tau = 0.70
        pos_mask = ((sim_matrix >= self.pos_tau) & (untrusted_label_1_mask | untrusted_label_2_mask)).float()
        neg_mask = ((sim_matrix < self.neg_tau) & untrusted_label_0_mask).float()
        mask = pos_mask - self.neg_rate * neg_mask ###
        ##
        sim_matrix = sim_matrix.float()  / sim_matrix.abs().sum(1, keepdim=True)
        sim_matrix = sim_matrix * mask
        ctrr_loss = self.lamb * (contrast_loss * sim_matrix).sum(dim=1).mean(0)
        # debug
        if torch.rand(1).item() < 0.005 and mask.sum() > 0:
            print('--------------------------------')
            # print(sim_matrix)
            print(f"mask.abs().sum(): {mask.abs().sum()}")
            print(f"stat_label_matrix: count_0: {(label_matrix == 0).sum()}, count_1: {(label_matrix == 1).sum()}, count_2: {(label_matrix == 2).sum()}, count_3: {(label_matrix == 3).sum()}")
            print(f"stat_age_matrix: count_0: {(age_matrix == 0).sum()}, count_1: {(age_matrix == 1).sum()}, count_2: {(age_matrix == 2).sum()}, count_3: {(age_matrix == 3).sum()}")
            print(f'neg_count: {neg_mask.sum()}')
            print(f'pos_count: {pos_mask.sum()}')
            print('--------------------------------')

        return preds, torch.stack(filter_list, dim=0), weights, ctrr_loss
    
class elrUGP(nn.Module):
    def __init__(self, n_snps, n_genes, d_hidden=64, n_filters=8, gene_dropout=0, snp_dropout=0, n_gnn_layers=2):
        super().__init__()


# UGP_v1 with AgeAwareMLP1
class AgeUGP_v1(nn.Module):
    """UGP model with age-aware layer and adversarial training from AgeAwareMLP1"""
    def __init__(self, n_snps, n_genes, d_hidden=64, n_filters=8, gene_dropout=0, snp_dropout=0, use_adversarial=True, use_consist=True, use_cumulative_rate=False):
        super().__init__()
        self.n_filters = n_filters
        self.use_adversarial = use_adversarial
        self.use_consist = use_consist
        self.use_cumulative_rate = use_cumulative_rate
        
        self.filter_list = nn.ParameterList()
        for _ in range(n_filters):
            self.filter_list.append(nn.Parameter(torch.randn(n_snps)*0.001))
        
        self.feature_extractor = nn.Sequential(
            nn.Dropout(gene_dropout),
            nn.Linear(n_genes, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(d_hidden, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.predictor = nn.Linear(d_hidden, 1)
        self.age_predictor = nn.Sequential(
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )
        # Age-aware layer from AgeAwareMLP1
        self.age_layer = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        self.apply(lambda module: bert_init_params(module))
        with torch.no_grad():
            nn.init.normal_(self.age_layer[-1].weight, mean=0, std=0.01)
            self.age_layer[-1].bias.data.copy_(torch.tensor([0., 1.]))
        self.dropout = nn.Dropout(snp_dropout)

    def get_transition_matrix(self, age_norm, age_cumulative_norm=None):
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
        trans_matrix[:, 0, 0] = 1.0    # p00
        trans_matrix[:, 0, 1] = 0.0    # p01
        trans_matrix[:, 1, 0] = pos_probs[:, 0]  # p10
        trans_matrix[:, 1, 1] = pos_probs[:, 1]  # p11
        return trans_matrix
    
    def forward(self, snp, snp_ids, g, age=None, labels=None):
        batch_size = len(snp)
        sample_h = []
        snp = snp.reshape(batch_size,-1,1)
        snp_h_list = []
        for i in range(self.n_filters):
            snp_h_list.append(torch.einsum('bnd,n->bnd', self.dropout(snp), self.filter_list[i]))
        snp_h = torch.concatenate(snp_h_list, dim=-1)
        
        for i in range(batch_size): 
            g.ndata['h'] = torch.index_select(snp_h[i],0,snp_ids)
            sample_h.append(torch.mean(dgl.readout_nodes(g, 'h', op='sum'),dim=-1).reshape(-1))
        sample_h = torch.stack(sample_h, dim=0)
        
        features = self.feature_extractor(sample_h)
        original_logits = self.predictor(features)
        
        filter_list = []
        for i in range(self.n_filters):
            filter_list.append(self.filter_list[i])
        
        if age is None or labels is None:
            return original_logits#, torch.stack(filter_list, dim=0)
            
        if self.use_cumulative_rate:
            age_cumulative = generate_cumulative_ages(age, mask=labels)
            age_cumulative_norm = (age_cumulative - 40) / (70 - 40)
            
        age_norm = (age - 40) / (70 - 40)
        neg_mask = (1 - labels.float())
        consist_weight = 0.0 if not self.use_consist else 1.0   
        if self.use_adversarial:
            features_rev = GradientReverseLayer.apply(features, 0.01)
            age_pred = self.age_predictor(features_rev)
            age_weight = 0.5
            age_loss = F.l1_loss(age_pred.squeeze(), age_norm)
        else:
            age_weight = 0.0
            age_loss = 0.0
            
        trans_matrix = self.get_transition_matrix(age_norm, age_cumulative_norm) if self.use_cumulative_rate else self.get_transition_matrix(age_norm)
        original_probs = torch.sigmoid(original_logits)
        original_probs = torch.clamp(original_probs, 1e-7, 1-1e-7)
        
        neg_dist = torch.cat([1 - original_probs, original_probs], dim=1)
        updated_dist = torch.bmm(neg_dist.unsqueeze(1), trans_matrix)
        updated_probs = updated_dist.squeeze(1)[:, 1:2]
        
        assert len(updated_probs[labels==1]) > 0
        consistency_loss = F.binary_cross_entropy(updated_probs[labels==1], original_probs[labels==1])
        
        final_probs = (1 - neg_mask) * original_probs + neg_mask * updated_probs
        final_logits = torch.log(final_probs / (1 - final_probs + 1e-7))
        
        final_loss = consist_weight * consistency_loss + age_weight * age_loss
        
        return final_logits, original_logits, final_loss

# UGP_v1 with AgeAwareMLP2
class AgeUGP_v2(nn.Module):
    """UGP model with feature disentanglement for age-related features from AgeAwareMLP2"""
    def __init__(self, n_snps, n_genes, d_hidden=64, n_filters=8, gene_dropout=0, snp_dropout=0, use_ageloss=True, use_disentangle=True, use_consist=True, use_cumulative_rate=False):
        super().__init__()
        self.n_filters = n_filters
        self.use_ageloss = use_ageloss
        self.use_disentangle = use_disentangle
        self.use_consist = use_consist
        self.use_cumulative_rate = use_cumulative_rate
        
        self.feature_dim = 16
        self.main_dim = 15
        self.age_dim = 1
        
        self.filter_list = nn.ParameterList()
        for _ in range(n_filters):
            self.filter_list.append(nn.Parameter(torch.randn(n_snps)*0.001))
        
        self.feature_extractor = nn.Sequential(
            nn.Dropout(gene_dropout),
            nn.Linear(n_genes, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(d_hidden, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
        )
        
        self.main_head = nn.Sequential(
            nn.Linear(self.main_dim, 1)
        )
        
        self.age_predictor = nn.Sequential(
            nn.Linear(self.age_dim, 1),
            nn.Sigmoid()
        )
        
        self.age_layer = nn.Sequential(
            nn.Linear(self.age_dim + 1, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        self.apply(lambda module: bert_init_params(module))        
        with torch.no_grad():
            nn.init.normal_(self.age_layer[-1].weight, mean=0, std=0.01)
            self.age_layer[-1].bias.data.copy_(torch.tensor([0., 1.]))
        self.dropout = nn.Dropout(snp_dropout)
    
    def get_intermediate_features(self, sample_h):
        features = self.feature_extractor(sample_h)
        main_feat = features[:, :self.main_dim]
        age_feat = features[:, self.main_dim:]
        return main_feat, age_feat
        
    def get_transition_matrix(self, age_norm, age_feat, age_cumulative_norm=None):
        if age_cumulative_norm is not None:
            combined_input = torch.cat([age_cumulative_norm, age_feat], dim=1)
        else:
            combined_input = torch.cat([age_norm, age_feat], dim=1)

        age_ori = age_norm * (70 - 40) + 40
        pos_trans = self.age_layer(combined_input)
        pos_probs = F.softmax(pos_trans, dim=1)
        
        # debug
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
    
    def forward(self, snp, snp_ids, g, age=None, labels=None):
        batch_size = len(snp)
        sample_h = []
        snp = snp.reshape(batch_size,-1,1)
        snp_h_list = []
        for i in range(self.n_filters):
            snp_h_list.append(torch.einsum('bnd,n->bnd', self.dropout(snp), self.filter_list[i]))
        snp_h = torch.concatenate(snp_h_list, dim=-1)
        
        for i in range(batch_size): 
            g.ndata['h'] = torch.index_select(snp_h[i],0,snp_ids)
            sample_h.append(torch.mean(dgl.readout_nodes(g, 'h', op='sum'),dim=-1).reshape(-1))
        sample_h = torch.stack(sample_h, dim=0)
        main_feat, age_feat = self.get_intermediate_features(sample_h)
        original_logits = self.main_head(main_feat)
        
        if age is None or labels is None:
            return original_logits
        
        if self.use_cumulative_rate:
            age_cumulative = generate_cumulative_ages(age, mask=labels)
            age_cumulative_norm = (age_cumulative - 40) / (70 - 40)
        else:
            age_cumulative_norm = None
        
        age_norm = (age - 40) / (70 - 40)
        age_pred = self.age_predictor(age_feat)
        age_loss = F.l1_loss(age_pred.squeeze(), age_norm)
        
        trans_matrix = self.get_transition_matrix(age_norm, age_feat, age_cumulative_norm) if self.use_cumulative_rate else self.get_transition_matrix(age_norm, age_feat)
        original_probs = torch.sigmoid(original_logits)
        original_probs = torch.clamp(original_probs, 1e-7, 1-1e-7)
        
        neg_mask = (1 - labels.float())
        consist_weight = 0.0 if not self.use_consist else 1.0
        age_weight = 0.0 if not self.use_ageloss else 0.5
        age_loss = 0.0 if not self.use_ageloss else age_loss
        
        neg_dist = torch.cat([1 - original_probs, original_probs], dim=1)
        updated_dist = torch.bmm(neg_dist.unsqueeze(1), trans_matrix)
        updated_probs = updated_dist.squeeze(1)[:, 1:2]
        print(f"updated_probs: {updated_probs}")
        
        assert len(updated_probs[labels==1]) > 0
        consistency_loss = F.binary_cross_entropy(updated_probs[labels==1], original_probs[labels==1])
        
        final_probs = (1 - neg_mask) * original_probs + neg_mask * updated_probs
        final_logits = torch.log(final_probs / (1 - final_probs + 1e-7))
        
        final_loss = consist_weight * consistency_loss + age_weight * age_loss
        
        return final_logits, original_logits, final_loss
