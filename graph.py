import numpy as np
import torch
import pdb

class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
       
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        
        self.off_set_l = off_set_l
        
        self.uniform = uniform
        
        # Temporal masking support
        self.temporal_mask_cutoff = None
        
        # Node-specific masking for counterfactual analysis
        self.masked_nodes = set()  # Set of node IDs to exclude during sampling
        
    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0] 
        # pdb.set_trace()
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr]) 
            e_idx_l.extend([x[1] for x in curr]) 
            n_ts_l.extend([x[2] for x in curr]) 
            off_set_l.append(len(n_idx_l)) 

   

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, off_set_l
        
    def find_before(self, src_idx, cut_time): 
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l 
        node_ts_l = self.node_ts_l 
        edge_idx_l = self.edge_idx_l 
        off_set_l = self.off_set_l 
        
        # pdb.set_trace()
        neighbors_idx = node_idx_l[off_set_l[src_idx] : off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx] : off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx] : off_set_l[src_idx + 1]]
        
        # Apply temporal mask if set
        if self.temporal_mask_cutoff is not None:
            # First filter by temporal mask (global cutoff)
            mask_valid = neighbors_ts <= self.temporal_mask_cutoff
            neighbors_idx = neighbors_idx[mask_valid]
            neighbors_ts = neighbors_ts[mask_valid]
            neighbors_e_idx = neighbors_e_idx[mask_valid]
        
        # Apply node-specific mask if set (for counterfactual analysis)
        if len(self.masked_nodes) > 0:
            # Filter out masked nodes
            node_mask_valid = ~np.isin(neighbors_idx, list(self.masked_nodes))
            neighbors_idx = neighbors_idx[node_mask_valid]
            neighbors_ts = neighbors_ts[node_mask_valid]
            neighbors_e_idx = neighbors_e_idx[node_mask_valid]
        
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_e_idx, neighbors_ts, neighbors_e_idx

        # Then apply the standard temporal constraint (cut_time)
        left = 0
        right = len(neighbors_idx) - 1
        
        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid
                
        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right+1], neighbors_e_idx[:right+1], neighbors_ts[:right+1], neighbors_e_idx[:right+1]
        else:
            return neighbors_idx[:left+1], neighbors_e_idx[:left+1], neighbors_ts[:left+1], neighbors_e_idx[:left+1]

    def get_temporal_neighbor_with_masks(self, src_idx_l, cut_time_l, mask_time_l, num_neighbors=20):
        """
        Enhanced temporal neighbor finding with per-edge masking
        
        Params
        ------
        src_idx_l: List[int] - source node indices
        cut_time_l: List[float] - per-edge temporal cutoff (standard temporal constraint)
        mask_time_l: List[float] - per-edge temporal mask (global visibility cutoff)
        num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l) == len(mask_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32) 
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32) 
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32) 
        out_ngh_eind_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time, mask_time) in enumerate(zip(src_idx_l, cut_time_l, mask_time_l)):
            # Temporarily set mask for this specific edge
            original_mask = self.temporal_mask_cutoff
            self.temporal_mask_cutoff = mask_time
            
            ngh_idx, ngh_eidx, ngh_ts, ngh_eind = self.find_before(src_idx, cut_time)
            
            # Restore original mask
            self.temporal_mask_cutoff = original_mask
          
            if len(ngh_idx) > 0: 
                if self.uniform: 
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    out_ngh_eind_batch[i, :] = ngh_eind[sampled_idx]

                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                    out_ngh_eind_batch[i, :] = out_ngh_eind_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    ngh_eind = ngh_eind[:num_neighbors]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    assert(len(ngh_eind) <= num_neighbors)
                    
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i,  num_neighbors - len(ngh_eidx):] = ngh_eidx
                    out_ngh_eind_batch[i, num_neighbors - len(ngh_eind):] = ngh_eind

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_ngh_eind_batch

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Original temporal neighbor finding method (for backward compatibility)
        
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32) 
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32) 
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_eind_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts, ngh_eind = self.find_before(src_idx, cut_time)
          
            if len(ngh_idx) > 0: 
                if self.uniform: 
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    ngh_eind = ngh_eind[:num_neighbors]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    assert(len(ngh_eind) <= num_neighbors)

                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx
                    out_ngh_eind_batch[i, num_neighbors - len(ngh_eind):] = ngh_eind

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_ngh_eind_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        nodes, e_feats, timestamps, e_indices = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [nodes]
        eidx_records = [e_feats] # This was misnamed, should be e_feat_records
        t_records = [timestamps]
        e_ind_records = [e_indices]

        for _ in range(k - 1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_ngh_eind_batch = self.get_temporal_neighbor(ngn_node_est, ngn_t_est, num_neighbors)
            
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_eind_batch = out_ngh_eind_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
            e_ind_records.append(out_ngh_eind_batch)
            
        return node_records, eidx_records, t_records, e_ind_records

    def set_temporal_mask(self, cutoff_time):
        """Set a global temporal cutoff - only edges with timestamp <= cutoff_time are visible"""
        self.temporal_mask_cutoff = cutoff_time
        
    def clear_temporal_mask(self):
        """Remove temporal masking - all edges are visible"""
        self.temporal_mask_cutoff = None

    def set_node_mask(self, masked_nodes):
        """Set nodes to be excluded during neighbor sampling (for counterfactual analysis)"""
        self.masked_nodes = set(masked_nodes) if masked_nodes is not None else set()
        
    def clear_node_mask(self):
        """Clear node masking - all nodes are visible during sampling"""
        self.masked_nodes = set()

            

