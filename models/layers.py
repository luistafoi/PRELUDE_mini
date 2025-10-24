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
        self.node_types = node_types

        # RNN for aggregating neighbors of each type
        self.rnn_aggregators = nn.ModuleDict({
            str(nt): nn.RNN(in_dim, in_dim, batch_first=True) for nt in self.node_types
        })
        
        # Semantic-level attention
        # It calculates the importance of each neighbor type for a given central node type
        self.sem_att = nn.Linear(in_dim * 2, 1, bias=False)
        
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, current_embeds, neigh_embeds_by_type):
        """
        Args:
            current_embeds (Tensor): Embeddings of the central nodes. Shape: (batch_size, in_dim)
            neigh_embeds_by_type (dict): A dictionary where keys are node types and values are
                                         tensors of neighbor embeddings.
                                         Shape of each value: (batch_size, num_samples, in_dim)
        Returns:
            Tensor: Updated embeddings for the central nodes. Shape: (batch_size, out_dim)
        """
        agg_embeds_by_type = {}
        
        # Aggregate neighbors for each type using RNN
        for n_type, neigh_feats in neigh_embeds_by_type.items():
            # Pass neighbor features through RNN. We only care about the last hidden state.
            _, last_hidden = self.rnn_aggregators[str(n_type)](neigh_feats)
            agg_embeds_by_type[n_type] = last_hidden.squeeze(0) # Squeeze to remove the num_layers dimension
        
        # Semantic Attention
        # Create pairs of (center_node_embedding, aggregated_neighbor_embedding)
        # We also include a self-loop pair
        sem_att_inputs = [torch.cat((current_embeds, current_embeds), dim=1)]
        sem_att_inputs.extend([
            torch.cat((current_embeds, agg_embeds), dim=1) 
            for n_type, agg_embeds in agg_embeds_by_type.items()
        ])
        
        # Stack for parallel attention calculation
        sem_att_stack = torch.stack(sem_att_inputs, dim=1)
        
        # Calculate attention weights and apply softmax
        att_weights = self.sem_att(sem_att_stack).squeeze(-1)
        att_weights = torch.softmax(att_weights, dim=1).unsqueeze(-1)
        
        # Weighted sum of embeddings (including self-loop)
        embeds_to_combine = [current_embeds] + list(agg_embeds_by_type.values())
        embeds_stack = torch.stack(embeds_to_combine, dim=1)
        
        # The new embedding is the attention-weighted sum of the aggregated neighbors
        updated_embeds = torch.sum(att_weights * embeds_stack, dim=1)
        
        return self.dropout(self.act(updated_embeds))