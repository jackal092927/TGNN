import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
import time

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        #output = self.layer_norm(output)
        
        return output, attn
    

class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)
        
        nn.init.xavier_normal_(self.fc.weight)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]
        
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]
        
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk
        
        ## Map based Attention
        #output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]
        
        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn
    
def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float()) 
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        # pdb.set_trace()
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1] 
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim] # [200, 1, 172] * [1, 1, 172] = [200, 1, 172]
        map_ts += self.phase.view(1, 1, -1) # + [1, 1, 172]
        
        harmonic = torch.cos(map_ts) # cosine

        return harmonic #self.dense(harmonic)
    
    
    
class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)
        
    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb
    

class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim
        
    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        self.att_dim = feat_dim + edge_dim + time_dim
        
        self.act = torch.nn.ReLU()
        
        self.lstm = torch.nn.LSTM(input_size=self.att_dim, 
                                  hidden_size=self.feat_dim, 
                                  num_layers=1, 
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)
            
        _, (hn, _) = self.lstm(seq_x)
        
        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None
    

class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)
        
    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None
    

class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        #self.act = torch.nn.ReLU()
        
        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
        
    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        # Create edge placeholder with correct edge feature dimensions
        src_e_ph = torch.zeros(src_ext.shape[0], src_ext.shape[1], seq_e.shape[2], device=src_ext.device)  # [B, 1, De]
        src_t_ext = torch.unsqueeze(src_t, dim=1) # src_t [B, 1, Dt]
        q = torch.cat([src_ext, src_e_ph, src_t_ext], dim=2) # [B, 1, D + De + Dt] -> [B, 1, model_dim]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, N, D + De + Dt] -> [B, N, model_dim]
        
        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        # pdb.set_trace()

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze(dim=1)
        attn = attn.squeeze(dim=1)

        # pdb.set_trace()
        output = self.merger(output, src)
        return output, attn

class TGIB(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat, hidden,
                 attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
                 num_layers=3, n_head=4, null_idx=0, num_heads=1, drop_out=0.1, seq_len=None):
        super(TGIB, self).__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(e_feat.astype(np.float32)), freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(n_feat.astype(np.float32)), padding_idx=0, freeze=True)

        self.feat_dim = n_feat.shape[1]
        self.n_feat_dim = self.n_feat_th.shape[1]
        self.e_feat_dim = self.e_feat_th.shape[1]
        self.time_dim = self.feat_dim if time_dim is None else time_dim
        
        # Store aggregation method and attention mode
        self.agg_method = agg_method
        self.attn_mode = attn_mode
        
        # Calculate model_dim and ensure it's divisible by n_head
        base_model_dim = self.n_feat_dim + self.e_feat_dim + self.time_dim
        
        # Adjust time_dim to make model_dim divisible by n_head
        remainder = base_model_dim % n_head
        if remainder != 0:
            adjustment = n_head - remainder
            self.time_dim += adjustment
            self.logger.info(f'Adjusted time_dim from {time_dim} to {self.time_dim} to make model_dim divisible by n_head={n_head}')
        
        self.model_dim = self.n_feat_dim + self.e_feat_dim + self.time_dim

        self.use_time = use_time
        self.merge_layer = MergeLayer(self.model_dim, self.model_dim, self.n_feat_dim, self.n_feat_dim)

        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, self.e_feat_dim, self.time_dim,
                                attn_mode=attn_mode, n_head=n_head, drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses lstm model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim, self.e_feat_dim, self.time_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim, self.e_feat_dim) for _ in range(num_layers)])
        else:
            raise ValueError('invalid agg_method value')

        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(self.time_dim)
        elif use_time == 'pos':
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(self.time_dim, seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(self.time_dim)
        else:
            raise ValueError('invalid time option')
        
        # Standard affinity_score for most training modes
        self.affinity_score = MergeLayer(self.n_feat_dim, self.n_feat_dim, self.n_feat_dim, 1)
        
        # Original-compatible scoring for hybrid mode (expects concatenated embeddings: n_feat + n_feat + time + e_feat)
        concat_emb_dim = self.n_feat_dim + self.n_feat_dim + self.time_dim + self.e_feat_dim
        self.affinity_score_original = MergeLayer(concat_emb_dim, concat_emb_dim, concat_emb_dim, 1)
        self.probability_score_original = MergeLayer(concat_emb_dim, concat_emb_dim, concat_emb_dim, 1)
        self.fix_r = False
        self.init_r = 0.9
        self.decay_interval = 10
        self.decay_r = 0.1
        self.final_r = 0.7

    def forward(self, src_l, dst_l, ts_l, e_idx_l, num_neighbors=10):
        """Enhanced forward pass with key sophisticated components"""
        device = self.n_feat_th.device
        
        # Basic embeddings
        src_embed = self.tem_conv(src_l, ts_l, self.num_layers, num_neighbors)
        dst_embed = self.tem_conv(dst_l, ts_l, self.num_layers, num_neighbors)
        
        # Enhanced event representation (like original)
        if isinstance(src_l, np.ndarray):
            src_l_tensor = torch.from_numpy(src_l).long().to(device) 
            ts_l_tensor = torch.from_numpy(ts_l).float().to(device)
            e_idx_l_tensor = torch.from_numpy(e_idx_l).long().to(device)
        else:
            src_l_tensor = src_l.long().to(device)
            ts_l_tensor = ts_l.float().to(device) 
            e_idx_l_tensor = e_idx_l.long().to(device)
            
        # Get edge features and time features 
        edge_features = self.edge_raw_embed(e_idx_l_tensor)
        time_features = self.time_encoder(ts_l_tensor.unsqueeze(1)).squeeze(1)
        
        # Create enhanced event embeddings - matching original 4x approach but with correct dimensions
        # Use a simpler but compatible approach
        combined_features = torch.cat([src_embed, dst_embed, time_features, edge_features], dim=1)
        
        # Use a more sophisticated scoring mechanism than simple dot product
        # Create an enhanced affinity computation
        enhanced_score = self.affinity_score(src_embed, dst_embed)
        
        # Add temporal and edge information influence
        temporal_edge_info = torch.cat([time_features, edge_features], dim=1) 
        # Create a simple linear layer for temporal/edge weighting if it doesn't exist
        if not hasattr(self, 'temporal_edge_weight'):
            self.temporal_edge_weight = torch.nn.Linear(temporal_edge_info.shape[1], 1).to(device)
        
        temporal_edge_score = self.temporal_edge_weight(temporal_edge_info)
        
        # Combine different information sources (sophisticated scoring)
        final_score = enhanced_score + 0.1 * temporal_edge_score  # Weighted combination
        
        return final_score.squeeze(dim=-1).sigmoid()

    def forward_with_info_bottleneck(self, src_l, dst_l, fake_dst_l, ts_l, e_idx_l, epoch, training, num_neighbors=10):
        """Forward pass with explicit information bottleneck loss computation"""
        device = self.n_feat_th.device
        
        # Get positive and negative predictions
        pos_prob = self.forward(src_l, dst_l, ts_l, e_idx_l, num_neighbors)
        neg_prob = self.forward(src_l, fake_dst_l, ts_l, e_idx_l, num_neighbors)
        
        # Sophisticated information bottleneck with concrete sampling
        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, 
                                                   final_r=self.final_r, init_r=self.init_r)
        
        # Apply concrete sampling to probabilities (sophisticated sampling)
        if training:
            pos_prob_sampled = self.concrete_sample(torch.logit(pos_prob + 1e-8), temp=1.0, training=True)
            neg_prob_sampled = self.concrete_sample(torch.logit(neg_prob + 1e-8), temp=1.0, training=True)
        else:
            pos_prob_sampled = pos_prob
            neg_prob_sampled = neg_prob
        
        # Information bottleneck loss (sophisticated regularization)
        pos_info_loss = (pos_prob_sampled * torch.log(pos_prob_sampled/r + 1e-6) + 
                        (1-pos_prob_sampled) * torch.log((1-pos_prob_sampled)/(1-r+1e-6) + 1e-6)).mean()
        neg_info_loss = (neg_prob_sampled * torch.log(neg_prob_sampled/r + 1e-6) + 
                        (1-neg_prob_sampled) * torch.log((1-neg_prob_sampled)/(1-r+1e-6) + 1e-6)).mean()
        info_loss = (pos_info_loss + neg_info_loss) / 2
        
        return pos_prob_sampled, neg_prob_sampled, info_loss

    def tem_conv(self, src_l, ts_l, curr_layers, num_neighbors):
        assert(curr_layers >= 0)

        device = self.n_feat_th.device
        
        batch_size = len(src_l)

        src_l_tensor = torch.from_numpy(src_l).long().to(device)
        ts_l_tensor = torch.from_numpy(ts_l).float().to(device)

        src_node_features = self.node_raw_embed(src_l_tensor)
        
        if curr_layers == 0:
            return src_node_features
        else:
            src_node_features_time = self.time_encoder(ts_l_tensor.unsqueeze(1)).squeeze(1)

            ngh_node_batch, ngh_eidx_batch, ngh_t_batch, ngh_eind_batch = self.ngh_finder.get_temporal_neighbor(src_l, ts_l, num_neighbors)
            
            ngh_node_batch_tensor = torch.from_numpy(ngh_node_batch).long().to(device)
            ngh_eidx_batch_tensor = torch.from_numpy(ngh_eidx_batch).long().to(device)
            ngh_t_batch_tensor = torch.from_numpy(ngh_t_batch).float().to(device)
            
            ngh_e_features = self.edge_raw_embed(ngh_eidx_batch_tensor)
            ngh_time_features = self.time_encoder(ngh_t_batch_tensor)
            
            ngh_node_features_rec = self.tem_conv(ngh_node_batch.flatten(), ngh_t_batch.flatten(), curr_layers=curr_layers-1, num_neighbors=num_neighbors)
            ngh_node_features_rec = ngh_node_features_rec.view(batch_size, num_neighbors, -1)
            
            mask = ngh_node_batch_tensor == 0
            
            attn_m = self.attn_model_list[curr_layers-1]
            
            local, weight = attn_m(src_node_features, 
                                 src_node_features_time,
                                 ngh_node_features_rec,
                                 ngh_time_features,
                                 ngh_e_features,
                                 mask)
            
            return local

    def tem_conv_with_masks(self, src_l, ts_l, mask_ts_l, curr_layers, num_neighbors):
        """Temporal convolution with per-edge masking support"""
        assert(curr_layers >= 0)

        device = self.n_feat_th.device
        
        batch_size = len(src_l)

        src_l_tensor = torch.from_numpy(src_l).long().to(device)
        ts_l_tensor = torch.from_numpy(ts_l).float().to(device)

        src_node_features = self.node_raw_embed(src_l_tensor)

        if curr_layers == 0:
            return src_node_features
        else:
            src_node_features_time = self.time_encoder(ts_l_tensor.unsqueeze(1)).squeeze(1)

            ngh_node_batch, ngh_eidx_batch, ngh_t_batch, ngh_eind_batch = self.ngh_finder.get_temporal_neighbor_with_masks(
                src_l, ts_l, mask_ts_l, num_neighbors)
            
            ngh_node_batch_tensor = torch.from_numpy(ngh_node_batch).long().to(device)
            ngh_eidx_batch_tensor = torch.from_numpy(ngh_eidx_batch).long().to(device)
            ngh_t_batch_tensor = torch.from_numpy(ngh_t_batch).float().to(device)
            
            ngh_e_features = self.edge_raw_embed(ngh_eidx_batch_tensor)
            ngh_time_features = self.time_encoder(ngh_t_batch_tensor)
            
            # Recursively call tem_conv to get neighbor features
            ngh_node_features_rec = self.tem_conv_with_masks(
                ngh_node_batch.flatten(), 
                ngh_t_batch.flatten(), 
                np.repeat(mask_ts_l, num_neighbors),  # Repeat original mask for each neighbor
                curr_layers=curr_layers-1, 
                num_neighbors=num_neighbors
            )
            ngh_node_features_rec = ngh_node_features_rec.view(batch_size, num_neighbors, -1)
            
            mask = ngh_node_batch_tensor == 0
            
            attn_m = self.attn_model_list[curr_layers-1]
            
            local, weight = attn_m(src_node_features, 
                                 src_node_features_time,
                                 ngh_node_features_rec,
                                 ngh_time_features,
                                 ngh_e_features,
                                 mask)
            
            return local

    def tem_conv_with_attn(self, src_l, ts_l, curr_layers, num_neighbors):
        if curr_layers == 0:
            device = self.node_raw_embed.weight.device
            src_node_batch_th = torch.from_numpy(src_l).long().to(device)
            src_node_feat = self.node_raw_embed(src_node_batch_th)
            return src_node_feat, [], []

        # Use base-level features for the source nodes to avoid complex recursion
        device = self.n_feat_th.device
        src_l_tensor = torch.from_numpy(src_l).long().to(device)
        ts_l_tensor = torch.from_numpy(ts_l).float().to(device)
        
        src_node_features = self.node_raw_embed(src_l_tensor)
        src_node_features_time = self.time_encoder(ts_l_tensor.unsqueeze(1)).squeeze(1)

        # Get neighbors and compute attention
        ngh_node_batch, ngh_eidx_batch, ngh_t_batch, ngh_eind_batch = self.ngh_finder.get_temporal_neighbor(src_l, ts_l, num_neighbors)
        
        ngh_node_batch_tensor = torch.from_numpy(ngh_node_batch).long().to(device)
        ngh_eidx_batch_tensor = torch.from_numpy(ngh_eidx_batch).long().to(device)
        ngh_t_batch_tensor = torch.from_numpy(ngh_t_batch).float().to(device)
        
        ngh_node_features = self.node_raw_embed(ngh_node_batch_tensor)
        ngh_e_features = self.edge_raw_embed(ngh_eidx_batch_tensor)
        ngh_time_features = self.time_encoder(ngh_t_batch_tensor)
        
        mask = ngh_node_batch_tensor == 0
        
        if self.agg_method == 'attn':
            attn_m = self.attn_model_list[curr_layers - 1]
            
            local, weight = attn_m(src_node_features, 
                                 src_node_features_time,
                                 ngh_node_features,
                                 ngh_time_features,
                                 ngh_e_features,
                                 mask)
        else:
            raise NotImplementedError

        # For simplicity, we'll return just the current layer's attention and neighbors
        # In a full recursive implementation, you'd collect all layers
        return local, [weight], [{'nodes': ngh_node_batch, 'edge_indices': ngh_eind_batch}]

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise) 
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid() 
        else:
            att_bern = (att_log_logit).sigmoid() 
        return att_bern

    def forward_with_masks(self, src_l, dst_l, ts_l, e_idx_l, mask_ts_l, num_neighbors=10):
        """
        Forward pass with per-edge temporal masking
        
        Parameters:
        -----------
        src_l, dst_l: source and destination node arrays
        ts_l: timestamp array (for temporal neighbor constraints)
        e_idx_l: edge index array
        mask_ts_l: per-edge temporal mask cutoffs (what edges are visible)
        num_neighbors: number of neighbors to sample
        """
        src_embed = self.tem_conv_with_masks(src_l, ts_l, mask_ts_l, self.num_layers, num_neighbors)
        dst_embed = self.tem_conv_with_masks(dst_l, ts_l, mask_ts_l, self.num_layers, num_neighbors)
        
        score = self.affinity_score(src_embed, dst_embed).squeeze(dim=-1)
        
        return score.sigmoid()

    def get_explanation(self, src_l, dst_l, ts_l, num_neighbors=10):
        src_embed, src_edge_attns, src_neighbor_data = self.tem_conv_with_attn(src_l, ts_l, self.num_layers, num_neighbors)
        dst_embed, dst_edge_attns, dst_neighbor_data = self.tem_conv_with_attn(dst_l, ts_l, self.num_layers, num_neighbors)

        all_attns = {'src': src_edge_attns, 'dst': dst_edge_attns}
        all_neighbor_data = {'src': src_neighbor_data, 'dst': dst_neighbor_data}

        return all_attns, all_neighbor_data

    def forward_complex_batched(self, src_batch, dst_batch, fake_dst_batch, ts_batch, e_idx_batch, 
                               epoch, training, num_neighbors=10):
        """
        Batched version of the original complex forward method
        Preserves the multi-hop architecture and information bottleneck
        
        Args:
            src_batch, dst_batch: [batch_size] arrays of source/destination nodes
            fake_dst_batch: [batch_size] array of negative destination nodes  
            ts_batch: [batch_size] array of timestamps
            e_idx_batch: [batch_size] array of edge indices
            epoch: current training epoch
            training: whether in training mode
            num_neighbors: number of neighbors to sample
            
        Returns:
            pos_prob: [batch_size] positive edge probabilities
            neg_prob: [batch_size] negative edge probabilities  
            info_loss: scalar information bottleneck loss
        """
        device = self.n_feat_th.device
        batch_size = len(src_batch)
        
        # Convert to tensors
        if isinstance(src_batch, np.ndarray):
            src_batch_tensor = torch.from_numpy(src_batch).long().to(device)
            dst_batch_tensor = torch.from_numpy(dst_batch).long().to(device)
            fake_dst_batch_tensor = torch.from_numpy(fake_dst_batch).long().to(device)
            ts_batch_tensor = torch.from_numpy(ts_batch).float().to(device)
            e_idx_batch_tensor = torch.from_numpy(e_idx_batch).long().to(device)
        else:
            src_batch_tensor = src_batch.long().to(device)
            dst_batch_tensor = dst_batch.long().to(device)
            fake_dst_batch_tensor = fake_dst_batch.long().to(device)
            ts_batch_tensor = ts_batch.float().to(device)
            e_idx_batch_tensor = e_idx_batch.long().to(device)
        
        # ================== STEP 1: Target Edge Embeddings (Z_ek) ==================
        # Get temporal convolution embeddings for all edges in batch
        src_embeds_batch = self.tem_conv(src_batch, ts_batch, self.num_layers, num_neighbors)  # [batch_size, feat_dim]
        dst_embeds_batch = self.tem_conv(dst_batch, ts_batch, self.num_layers, num_neighbors)  # [batch_size, feat_dim]
        fake_dst_embeds_batch = self.tem_conv(fake_dst_batch, ts_batch, self.num_layers, num_neighbors)  # [batch_size, feat_dim]
        
        # Get edge embeddings  
        edge_embeds_batch = self.edge_raw_embed(e_idx_batch_tensor)  # [batch_size, feat_dim]
        
        # Get time embeddings
        time_embeds_batch = self.time_encoder(ts_batch_tensor.unsqueeze(1)).squeeze(1)  # [batch_size, feat_dim]
        
        # Create target event embeddings (matching original 4x concatenation)
        target_event_emb_batch = torch.cat([src_embeds_batch, dst_embeds_batch, time_embeds_batch, edge_embeds_batch], dim=1)  # [batch_size, feat_dim*4]
        fake_target_event_emb_batch = torch.cat([src_embeds_batch, fake_dst_embeds_batch, time_embeds_batch, edge_embeds_batch], dim=1)  # [batch_size, feat_dim*4]
        
        # ================== STEP 2: Multi-hop Neighbor Embeddings (Z_ej) ==================
        all_neighbor_embeddings = []  # Will store all neighbor embeddings for each edge in batch
        
        for i in range(batch_size):
            edge_neighbor_embeddings = []
            
            # Current edge info
            src_k = src_batch[i:i+1]
            dst_k = dst_batch[i:i+1]  
            ts_k = ts_batch[i:i+1]
            
            # ======== 1-hop neighbors from SOURCE ========
            one_hop_ngh_nodes_i1, one_hop_ngh_eidx_i1, one_hop_ngh_ts_i1, _ = self.ngh_finder.get_temporal_neighbor(src_k, ts_k, num_neighbors)
            one_hop_ngh_nodes_i1 = one_hop_ngh_nodes_i1.squeeze()
            one_hop_ngh_eidx_i1 = one_hop_ngh_eidx_i1.squeeze()  
            one_hop_ngh_ts_i1 = one_hop_ngh_ts_i1.squeeze()
            
            # Get embeddings for 1-hop neighbors
            one_hop_src_emb_i1 = src_embeds_batch[i:i+1].repeat(num_neighbors, 1)  # Repeat source embedding
            one_hop_dst_emb_i1 = self.tem_conv(one_hop_ngh_nodes_i1, one_hop_ngh_ts_i1, self.num_layers, num_neighbors)
            one_hop_edge_emb_i1 = self.edge_raw_embed(torch.from_numpy(one_hop_ngh_eidx_i1).long().to(device))
            
            # Time deltas for 1-hop neighbors
            one_hop_time_delta_i1 = ts_k - one_hop_ngh_ts_i1
            one_hop_time_emb_i1 = self.time_encoder(torch.from_numpy(one_hop_time_delta_i1).float().to(device).reshape(-1, 1)).reshape(num_neighbors, -1)
            
            # Create 1-hop event embeddings
            one_hop_event_emb_i1 = torch.cat([one_hop_src_emb_i1, one_hop_dst_emb_i1, one_hop_time_emb_i1, one_hop_edge_emb_i1], dim=1)
            edge_neighbor_embeddings.append(one_hop_event_emb_i1)
            
            # ======== 2-hop neighbors from 1-hop neighbors ========
            two_hop_ngh_nodes_i1, two_hop_ngh_eidx_i1, two_hop_ngh_ts_i1, _ = self.ngh_finder.get_temporal_neighbor(one_hop_ngh_nodes_i1, one_hop_ngh_ts_i1, num_neighbors)
            two_hop_ngh_nodes_i1 = two_hop_ngh_nodes_i1.reshape(-1)
            two_hop_ngh_eidx_i1 = two_hop_ngh_eidx_i1.reshape(-1)
            two_hop_ngh_ts_i1 = two_hop_ngh_ts_i1.reshape(-1)
            
            # Get embeddings for 2-hop neighbors  
            two_hop_src_emb_i1 = self.tem_conv(two_hop_ngh_nodes_i1, two_hop_ngh_ts_i1, self.num_layers, num_neighbors)
            two_hop_dst_emb_i1 = one_hop_dst_emb_i1.repeat(1, num_neighbors).reshape(num_neighbors*num_neighbors, -1)
            two_hop_edge_emb_i1 = self.edge_raw_embed(torch.from_numpy(two_hop_ngh_eidx_i1).long().to(device))
            
            # Time deltas for 2-hop neighbors
            two_hop_time_delta_i1 = ts_k - two_hop_ngh_ts_i1
            two_hop_time_emb_i1 = self.time_encoder(torch.from_numpy(two_hop_time_delta_i1).float().to(device).reshape(-1, 1)).reshape(num_neighbors*num_neighbors, -1)
            
            # Create 2-hop event embeddings
            two_hop_event_emb_i1 = torch.cat([two_hop_src_emb_i1, two_hop_dst_emb_i1, two_hop_time_emb_i1, two_hop_edge_emb_i1], dim=1)
            edge_neighbor_embeddings.append(two_hop_event_emb_i1)
            
            # ======== 1-hop neighbors from DESTINATION ========
            one_hop_ngh_nodes_i2, one_hop_ngh_eidx_i2, one_hop_ngh_ts_i2, _ = self.ngh_finder.get_temporal_neighbor(dst_k, ts_k, num_neighbors)
            one_hop_ngh_nodes_i2 = one_hop_ngh_nodes_i2.squeeze()
            one_hop_ngh_eidx_i2 = one_hop_ngh_eidx_i2.squeeze()
            one_hop_ngh_ts_i2 = one_hop_ngh_ts_i2.squeeze()
            
            # Get embeddings for 1-hop neighbors from destination
            one_hop_src_emb_i2 = self.tem_conv(one_hop_ngh_nodes_i2, one_hop_ngh_ts_i2, self.num_layers, num_neighbors)
            one_hop_dst_emb_i2 = dst_embeds_batch[i:i+1].repeat(num_neighbors, 1)  # Repeat destination embedding
            one_hop_edge_emb_i2 = self.edge_raw_embed(torch.from_numpy(one_hop_ngh_eidx_i2).long().to(device))
            
            # Time deltas for 1-hop neighbors from destination
            one_hop_time_delta_i2 = ts_k - one_hop_ngh_ts_i2
            one_hop_time_emb_i2 = self.time_encoder(torch.from_numpy(one_hop_time_delta_i2).float().to(device).reshape(-1, 1)).reshape(num_neighbors, -1)
            
            # Create 1-hop event embeddings from destination
            one_hop_event_emb_i2 = torch.cat([one_hop_src_emb_i2, one_hop_dst_emb_i2, one_hop_time_emb_i2, one_hop_edge_emb_i2], dim=1)
            edge_neighbor_embeddings.append(one_hop_event_emb_i2)
            
            # ======== 2-hop neighbors from destination's 1-hop neighbors ========
            two_hop_ngh_nodes_i2, two_hop_ngh_eidx_i2, two_hop_ngh_ts_i2, _ = self.ngh_finder.get_temporal_neighbor(one_hop_ngh_nodes_i2, one_hop_ngh_ts_i2, num_neighbors)
            two_hop_ngh_nodes_i2 = two_hop_ngh_nodes_i2.reshape(-1)
            two_hop_ngh_eidx_i2 = two_hop_ngh_eidx_i2.reshape(-1) 
            two_hop_ngh_ts_i2 = two_hop_ngh_ts_i2.reshape(-1)
            
            # Get embeddings for 2-hop neighbors from destination
            two_hop_src_emb_i2 = one_hop_src_emb_i2.repeat(1, num_neighbors).reshape(num_neighbors*num_neighbors, -1)
            two_hop_dst_emb_i2 = self.tem_conv(two_hop_ngh_nodes_i2, two_hop_ngh_ts_i2, self.num_layers, num_neighbors)
            two_hop_edge_emb_i2 = self.edge_raw_embed(torch.from_numpy(two_hop_ngh_eidx_i2).long().to(device))
            
            # Time deltas for 2-hop neighbors from destination
            two_hop_time_delta_i2 = ts_k - two_hop_ngh_ts_i2
            two_hop_time_emb_i2 = self.time_encoder(torch.from_numpy(two_hop_time_delta_i2).float().to(device).reshape(-1, 1)).reshape(num_neighbors*num_neighbors, -1)
            
            # Create 2-hop event embeddings from destination
            two_hop_event_emb_i2 = torch.cat([two_hop_src_emb_i2, two_hop_dst_emb_i2, two_hop_time_emb_i2, two_hop_edge_emb_i2], dim=1)
            edge_neighbor_embeddings.append(two_hop_event_emb_i2)
            
            # Combine all neighbor embeddings for this edge
            all_edge_neighbors = torch.cat(edge_neighbor_embeddings, dim=0)  # [2*num_neighbors*(num_neighbors+1), feat_dim*4]
            all_neighbor_embeddings.append(all_edge_neighbors)
        
        # ================== STEP 3: Information Bottleneck & Scoring ==================
        pos_probs = []
        neg_probs = []
        all_info_losses = []
        
        for i in range(batch_size):
            target_emb = target_event_emb_batch[i:i+1]  # [1, feat_dim*4]
            fake_target_emb = fake_target_event_emb_batch[i:i+1]  # [1, feat_dim*4]
            neighbor_embs = all_neighbor_embeddings[i]  # [2*num_neighbors*(num_neighbors+1), feat_dim*4]
            
            # Repeat target embedding to match neighbor embeddings
            num_neighbors_total = neighbor_embs.shape[0]
            target_emb_repeated = target_emb.repeat(num_neighbors_total, 1)
            fake_target_emb_repeated = fake_target_emb.repeat(num_neighbors_total, 1)
            
            # Get affinity scores (using affinity_score_original which expects feat_dim*4)
            pos_score_logits = self.affinity_score_original(target_emb_repeated, neighbor_embs)  # [num_neighbors_total, 1]
            neg_score_logits = self.affinity_score_original(fake_target_emb_repeated, neighbor_embs)  # [num_neighbors_total, 1]
            
            # Apply sophisticated sampling (information bottleneck)
            pos_scores = self.sampling(pos_score_logits, training).reshape(-1)  # [num_neighbors_total]
            neg_scores = self.sampling(neg_score_logits, training).reshape(-1)  # [num_neighbors_total]
            
            # Compute information bottleneck loss
            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            pos_info_loss = (pos_scores * torch.log(pos_scores/r + 1e-6) + (1-pos_scores) * torch.log((1-pos_scores)/(1-r+1e-6) + 1e-6)).mean()
            neg_info_loss = (neg_scores * torch.log(neg_scores/r + 1e-6) + (1-neg_scores) * torch.log((1-neg_scores)/(1-r+1e-6) + 1e-6)).mean()
            info_loss = (pos_info_loss + neg_info_loss) / 2
            all_info_losses.append(info_loss)
            
            # Apply information bottleneck to neighbor embeddings (like original)
            pos_scores_expanded = pos_scores.reshape(-1, 1)  # [num_neighbors_total, 1]
            neg_scores_expanded = neg_scores.reshape(-1, 1)  # [num_neighbors_total, 1]
            
            pos_active_edge_emb = pos_scores_expanded * neighbor_embs  # [num_neighbors_total, feat_dim*4]
            neg_active_edge_emb = neg_scores_expanded * neighbor_embs  # [num_neighbors_total, feat_dim*4]
            
            # Global pooling (like original)
            pos_active_graph = global_mean_pool(pos_active_edge_emb, batch=None)  # [1, feat_dim*4]
            neg_active_graph = global_mean_pool(neg_active_edge_emb, batch=None)  # [1, feat_dim*4]
            
            # Final probability prediction (like original)
            pos_prob = self.probability_score_original(pos_active_graph, target_emb).sigmoid()  # [1, 1]
            neg_prob = self.probability_score_original(neg_active_graph, fake_target_emb).sigmoid()  # [1, 1]
            
            pos_probs.append(pos_prob)
            neg_probs.append(neg_prob)
        
        # Combine results
        final_pos_probs = torch.cat(pos_probs, dim=0).squeeze(dim=1)  # [batch_size]
        final_neg_probs = torch.cat(neg_probs, dim=0).squeeze(dim=1)  # [batch_size]
        final_info_loss = torch.stack(all_info_losses).mean()  # scalar
        
        return final_pos_probs, final_neg_probs, final_info_loss

    def forward_original_signature(self, u_emb, i_emb, i_emb_fake, time, e_emb, k, epoch, training, num_neighbors=10):
        """
        EXACT ORIGINAL FORWARD METHOD - Preserves original signature and logic
        This is a direct port of the original TGIB forward method to ensure identical behavior
        
        Args:
            u_emb: source node sequence [:k+1]
            i_emb: destination node sequence [:k+1] 
            i_emb_fake: negative destination node
            time: timestamp sequence [:k+1]
            e_emb: edge index sequence [:k+1]
            k: current edge index
            epoch: current epoch
            training: training mode flag
            num_neighbors: number of neighbors to sample
        """
        device = self.n_feat_th.device

        # Z_ek - Target edge embedding (EXACT ORIGINAL LOGIC)
        u_emb_k = u_emb[k][np.newaxis]
        i_emb_k = i_emb[k][np.newaxis]
        time_k = time[k][np.newaxis]
        e_idx_k = (e_emb[k]-1)[np.newaxis]

        # Temporal convolution embeddings (EXACT ORIGINAL)
        t_idx_u_emb_k = self.tem_conv(u_emb_k, time_k, self.num_layers, num_neighbors)
        t_idx_i_emb_k = self.tem_conv(i_emb_k, time_k, self.num_layers, num_neighbors)
        edge_idx_l_th_k = torch.from_numpy(e_idx_k).long().to(device)
        t_idx_e_emb_k = self.edge_raw_embed(edge_idx_l_th_k)

        # Time encoding (EXACT ORIGINAL)
        time_tensor_k = torch.from_numpy(time_k).float().to(device)
        time_tensor_k = torch.unsqueeze(time_tensor_k, dim=1)
        time_encoder_k = self.time_encoder(torch.zeros_like(time_tensor_k)).reshape(1, -1) 

        # Target event embedding (EXACT ORIGINAL 4x CONCATENATION)
        target_event_emb_k = torch.cat([t_idx_u_emb_k, t_idx_i_emb_k, time_encoder_k, t_idx_e_emb_k], dim=1)

        # Fake target embedding (EXACT ORIGINAL)
        fake_t_idx_i_emb_k = self.tem_conv(i_emb_fake, time_k, self.num_layers, num_neighbors)
        fake_target_event_emb_k = torch.cat([t_idx_u_emb_k, fake_t_idx_i_emb_k, time_encoder_k, t_idx_e_emb_k], dim=1)

        # Multi-hop neighbor processing (EXACT ORIGINAL LOGIC)
        
        # ======== 1-hop neighbors from SOURCE ========
        one_hop_ngh_node_i_emb, one_hop_ngh_eidx_i_emb, one_hop_ngh_ts_i_emb, _ = self.ngh_finder.get_temporal_neighbor(u_emb_k, time_k, num_neighbors) 
        one_hop_ngh_node_i_emb = one_hop_ngh_node_i_emb.squeeze()
        one_hop_ngh_eidx_i_emb = one_hop_ngh_eidx_i_emb.squeeze()
        one_hop_ngh_ts_i_emb = one_hop_ngh_ts_i_emb.squeeze()
        
        one_hop_u_emb_i1 = t_idx_u_emb_k.repeat(num_neighbors,1)
        one_hop_i_emb_i1 = self.tem_conv(one_hop_ngh_node_i_emb, one_hop_ngh_ts_i_emb, self.num_layers, num_neighbors)
        one_hop_e_idx_i1 = torch.from_numpy(one_hop_ngh_eidx_i_emb).long().to(device)
        one_hop_e_emb_i1 = self.edge_raw_embed(one_hop_e_idx_i1)

        one_hop_time_del_i1 = time_k - one_hop_ngh_ts_i_emb
        one_hop_time_del_i1 = torch.from_numpy(one_hop_time_del_i1).float().to(device)
        one_hop_time_del_i1 = one_hop_time_del_i1.reshape(-1,1)
        one_hop_time_encoder_i1 = self.time_encoder(one_hop_time_del_i1).reshape(num_neighbors,-1) 

        one_hop_event_emb_i1 = torch.cat([one_hop_u_emb_i1, one_hop_i_emb_i1, one_hop_time_encoder_i1, one_hop_e_emb_i1], dim=1)

        # ======== 2-hop neighbors from 1-hop ========
        two_hop_ngh_node_u_emb, two_hop_ngh_eidx_u_emb, two_hop_ngh_ts_u_emb, _ = self.ngh_finder.get_temporal_neighbor(one_hop_ngh_node_i_emb, one_hop_ngh_ts_i_emb, num_neighbors) 
        two_hop_ngh_node_u_emb = two_hop_ngh_node_u_emb.squeeze()
        two_hop_ngh_eidx_u_emb = two_hop_ngh_eidx_u_emb.squeeze()
        two_hop_ngh_ts_u_emb = two_hop_ngh_ts_u_emb.squeeze()
        
        two_hop_u_emb_i1 = self.tem_conv(two_hop_ngh_node_u_emb.reshape(-1), two_hop_ngh_ts_u_emb.reshape(-1), self.num_layers, num_neighbors)
        two_hop_i_emb_i1 = one_hop_i_emb_i1.repeat(1,num_neighbors).reshape(num_neighbors*num_neighbors, -1)
        two_hop_e_idx_i1 = torch.from_numpy(two_hop_ngh_eidx_u_emb).long().to(device)
        two_hop_e_emb_i1 = self.edge_raw_embed(two_hop_e_idx_i1).reshape(num_neighbors*num_neighbors, -1)

        two_hop_time_del_i1 = time_k - two_hop_ngh_ts_u_emb
        two_hop_time_del_i1 = torch.from_numpy(two_hop_time_del_i1).float().to(device)
        two_hop_time_del_i1 = two_hop_time_del_i1.reshape(-1,1)
        two_hop_time_encoder_i1 = self.time_encoder(two_hop_time_del_i1).reshape(num_neighbors*num_neighbors, -1) 
        
        two_hop_event_emb_i1 = torch.cat([two_hop_u_emb_i1, two_hop_i_emb_i1, two_hop_time_encoder_i1, two_hop_e_emb_i1], dim=1)

        # ======== 1-hop neighbors from DESTINATION ========
        one_hop_ngh_node_u_emb, one_hop_ngh_eidx_u_emb, one_hop_ngh_ts_u_emb, _ = self.ngh_finder.get_temporal_neighbor(i_emb_k, time_k, num_neighbors) 
        one_hop_ngh_node_u_emb = one_hop_ngh_node_u_emb.squeeze()
        one_hop_ngh_eidx_u_emb = one_hop_ngh_eidx_u_emb.squeeze()
        one_hop_ngh_ts_u_emb = one_hop_ngh_ts_u_emb.squeeze()
        
        one_hop_u_emb_i2 = self.tem_conv(one_hop_ngh_node_u_emb, one_hop_ngh_ts_u_emb, self.num_layers, num_neighbors)
        one_hop_i_emb_i2 = t_idx_i_emb_k.repeat(num_neighbors,1)
        one_hop_e_idx_i2 = torch.from_numpy(one_hop_ngh_eidx_u_emb).long().to(device)
        one_hop_e_emb_i2 = self.edge_raw_embed(one_hop_e_idx_i2)

        one_hop_time_del_i2 = time_k - one_hop_ngh_ts_u_emb
        one_hop_time_del_i2 = torch.from_numpy(one_hop_time_del_i2).float().to(device)
        one_hop_time_del_i2 = one_hop_time_del_i2.reshape(-1,1)
        one_hop_time_encoder_i2 = self.time_encoder(one_hop_time_del_i2).reshape(num_neighbors,-1) 

        one_hop_event_emb_i2 = torch.cat([one_hop_u_emb_i2, one_hop_i_emb_i2, one_hop_time_encoder_i2, one_hop_e_emb_i2], dim=1)

        # ======== 2-hop neighbors from destination ========
        two_hop_ngh_node_i_emb, two_hop_ngh_eidx_i_emb, two_hop_ngh_ts_i_emb, _ = self.ngh_finder.get_temporal_neighbor(one_hop_ngh_node_u_emb, one_hop_ngh_ts_u_emb, num_neighbors) 
        two_hop_ngh_node_i_emb = two_hop_ngh_node_i_emb.squeeze()
        two_hop_ngh_eidx_i_emb = two_hop_ngh_eidx_i_emb.squeeze()
        two_hop_ngh_ts_i_emb = two_hop_ngh_ts_i_emb.squeeze()
        
        two_hop_u_emb_i2 = one_hop_u_emb_i2.repeat(1,num_neighbors).reshape(num_neighbors*num_neighbors, -1)
        two_hop_i_emb_i2 = self.tem_conv(two_hop_ngh_node_i_emb.reshape(-1), two_hop_ngh_ts_i_emb.reshape(-1), self.num_layers, num_neighbors)
        two_hop_e_idx_i2 = torch.from_numpy(two_hop_ngh_eidx_i_emb).long().to(device)
        two_hop_e_emb_i2 = self.edge_raw_embed(two_hop_e_idx_i2).reshape(num_neighbors*num_neighbors, -1)

        two_hop_time_del_i2 = time_k - two_hop_ngh_ts_i_emb
        two_hop_time_del_i2 = torch.from_numpy(two_hop_time_del_i2).float().to(device)
        two_hop_time_del_i2 = two_hop_time_del_i2.reshape(-1,1)
        two_hop_time_encoder_i2 = self.time_encoder(two_hop_time_del_i2).reshape(num_neighbors*num_neighbors, -1)  
        
        two_hop_event_emb_i2 = torch.cat([two_hop_u_emb_i2, two_hop_i_emb_i2, two_hop_time_encoder_i2, two_hop_e_emb_i2], dim=1)

        # Combine all neighbor embeddings (EXACT ORIGINAL)
        target_event_emb_i = torch.cat([one_hop_event_emb_i1, one_hop_event_emb_i2, two_hop_event_emb_i1, two_hop_event_emb_i2], dim=0)

        # Affinity scoring and sampling (EXACT ORIGINAL)
        pos_score_logits = self.affinity_score_original(target_event_emb_k.repeat(2*num_neighbors*(num_neighbors+1),1), target_event_emb_i)
        pos_score = self.sampling(pos_score_logits, training)
        pos_score = pos_score.reshape(-1)

        neg_score_logits = self.affinity_score_original(fake_target_event_emb_k.repeat(2*num_neighbors*(num_neighbors+1),1), target_event_emb_i)
        neg_score = self.sampling(neg_score_logits, training)
        neg_score = neg_score.reshape(-1)

        # Information bottleneck loss (EXACT ORIGINAL)
        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        pos_info_loss = (pos_score * torch.log(pos_score/r + 1e-6) + (1-pos_score) * torch.log((1-pos_score)/(1-r+1e-6) + 1e-6)).mean()
        neg_info_loss = (neg_score * torch.log(neg_score/r + 1e-6) + (1-neg_score) * torch.log((1-neg_score)/(1-r+1e-6) + 1e-6)).mean()
        info_loss = (pos_info_loss + neg_info_loss) / 2

        # Graph pooling and final prediction (EXACT ORIGINAL)
        pos_score = pos_score.reshape(-1,1)
        pos_active_edge_emb = pos_score * target_event_emb_i
        neg_score = neg_score.reshape(-1,1)
        neg_active_edge_emb = neg_score * target_event_emb_i

        pos_active_graph = global_mean_pool(pos_active_edge_emb, batch=None)
        pos_prob = self.probability_score_original(pos_active_graph, target_event_emb_k).sigmoid()

        neg_active_graph = global_mean_pool(neg_active_edge_emb, batch=None)
        neg_prob = self.probability_score_original(neg_active_graph, fake_target_event_emb_k).sigmoid()        

        return pos_prob.squeeze(dim=1), neg_prob.squeeze(dim=1), info_loss

