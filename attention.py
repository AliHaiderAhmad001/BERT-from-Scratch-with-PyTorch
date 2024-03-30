import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    """
    Attention head implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        head_dim (int): Dimensionality of the attention head.

    Attributes:
        query_weights (nn.Linear): Linear layer for query projection.
        key_weights (nn.Linear): Linear layer for key projection.
        value_weights (nn.Linear): Linear layer for value projection.
    """

    def __init__(self, hidden_size, head_dim):
        """
        Initializes the AttentionHead.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            head_dim (int): Dimensionality of the attention head.
        """
        super().__init__()
        self.head_dim = head_dim
        self.query_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, training:bool = False) -> torch.Tensor:
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated value embeddings after applying attention mechanism.
        """
        query: torch.Tensor = self.query_weights(query)
        key: torch.Tensor = self.key_weights(key)
        value: torch.Tensor = self.value_weights(value)

        att_scores: torch.Tensor = torch.matmul(query, key.transpose(1, 2)) / self.head_dim ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores: torch.Tensor = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)
        att_weights: torch.Tensor = self.dropout(att_weights, training = training) 
        n_value: torch.Tensor = torch.matmul(att_weights, value)

        return n_value


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer implementation.

    Args:
        config (object): Configuration object containing hyperparameters.
            - hidden_size (int): Hidden size for the model (embedding dimension).
            - num_heads (int): Number of attention heads.
            - head_dim (int): Dimensionality of each attention head.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        attention_heads (nn.ModuleList): List of AttentionHead layers.
        fc (nn.Linear): Fully connected layer for final projection.
    """

    def __init__(self, config):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            config (object): Configuration object containing hyperparameters.
                - hidden_size (int): Hidden size for the model (embedding dimension).
                - num_heads (int): Number of attention heads.
                - head_dim (int): Dimensionality of each attention head.
        """
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.head_dim: int = config.hidden_size // config.num_heads
        self.attention_heads: nn.ModuleList = nn.ModuleList([AttentionHead(self.hidden_size, self.head_dim) for _ in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, training:bool = False) -> torch.Tensor:
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask, training = training) for attention_head in self.attention_heads]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state
