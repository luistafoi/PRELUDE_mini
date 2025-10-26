# models/layers.py

import torch
import torch.nn as nn
import random

class RnnGnnLayer(nn.Module):
    """
    A single layer of the Heterogeneous GNN that uses an RNN for neighbor aggregation
    and semantic attention to combine information from different node types.
    """
    def __init__(self, in_dim, out_dim, node_types, dropout_rate=0.4):
        super(RnnGnnLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_types = node_types # Expecting list/set of integer type IDs

        # RNN for aggregating neighbors of each type
        self.rnn_aggregators = nn.ModuleDict({
            # Use string keys for ModuleDict
            str(nt): nn.RNN(in_dim, in_dim, batch_first=True) for nt in self.node_types
        })
        
        # Semantic-level attention
        self.sem_att = nn.Linear(in_dim * 2, 1, bias=False)
        
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
         # Optional: Add weight initialization if desired
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier Uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                     nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RNN):
                 for name, param in module.named_parameters():
                     if 'weight_ih' in name:
                         nn.init.xavier_uniform_(param.data)
                     elif 'weight_hh' in name:
                         nn.init.orthogonal_(param.data)
                     elif 'bias' in name:
                         param.data.fill_(0)


    def forward(self, current_embeds, neigh_embeds_by_type):
        """
        Args:
            current_embeds (Tensor): Embeddings of the central nodes. Shape: (batch_size, in_dim)
            neigh_embeds_by_type (dict): Keys are node type IDs (int), values are
                                         tensors of neighbor embeddings.
                                         Shape of each value: (batch_size, num_samples, in_dim)
        Returns:
            Tensor: Updated embeddings for the central nodes. Shape: (batch_size, out_dim)
        """
        agg_embeds_by_type = {}
        
        # Aggregate neighbors for each type using RNN
        for n_type, neigh_feats in neigh_embeds_by_type.items():
            if neigh_feats.nelement() > 0: # Check if tensor is not empty
                # Pass neighbor features through RNN. We only care about the last hidden state.
                _, last_hidden = self.rnn_aggregators[str(n_type)](neigh_feats)
                agg_embeds_by_type[n_type] = last_hidden.squeeze(0) # Squeeze batch dim
            else:
                # Handle cases with no neighbors of this type: use zero vector
                agg_embeds_by_type[n_type] = torch.zeros_like(current_embeds)
        
        # Semantic Attention
        # Create pairs of (center_node_embedding, aggregated_neighbor_embedding)
        sem_att_inputs = [torch.cat((current_embeds, current_embeds), dim=1)] # Self-loop
        sem_att_keys_ordered = sorted(agg_embeds_by_type.keys()) # Ensure consistent order
        sem_att_inputs.extend([
            torch.cat((current_embeds, agg_embeds_by_type[nt]), dim=1) 
            for nt in sem_att_keys_ordered
        ])
        
        # Stack for parallel attention calculation
        sem_att_stack = torch.stack(sem_att_inputs, dim=1)
        
        # Calculate attention weights and apply softmax
        att_weights = self.sem_att(sem_att_stack).squeeze(-1) # Shape: (batch_size, num_types + 1)
        att_weights = torch.softmax(att_weights, dim=1).unsqueeze(-1) # Shape: (batch_size, num_types + 1, 1)
        
        # Weighted sum of embeddings (including self-loop)
        embeds_to_combine = [current_embeds] + [agg_embeds_by_type[nt] for nt in sem_att_keys_ordered]
        embeds_stack = torch.stack(embeds_to_combine, dim=1) # Shape: (batch_size, num_types + 1, in_dim)
        
        # The new embedding is the attention-weighted sum
        updated_embeds = torch.sum(att_weights * embeds_stack, dim=1) # Shape: (batch_size, in_dim)
        
        # Apply activation and dropout
        # Note: If in_dim != out_dim, a Linear layer would be needed here. Assumed in_dim == out_dim
        return self.dropout(self.act(updated_embeds))