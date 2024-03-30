import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward

class Encoder(nn.Module):
    """
    Encoder layer implementation.

    Args:
        config (object): Configuration object containing hyperparameters.
            - hidden_size (int): Hidden size for the model (embedding dimension).
            - hidden_dropout_prob (float): Dropout probability for regularization.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        hidden_dropout_prob (float): Dropout probability for regularization.
        multihead_attention (MultiHeadAttention): Multi-head attention layer.
        norm1 (nn.LayerNorm): Layer normalization layer.
        norm2 (nn.LayerNorm): Layer normalization layer.
        feed_forward (FeedForward): Feed-forward layer.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, config):
        """
        Initializes the Encoder layer.

        Args:
            config (object): Configuration object containing hyperparameters.
                - hidden_size (int): Hidden size for the model (embedding dimension).
                - hidden_dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()

        self.hidden_size: int = config.hidden_size
        self.hidden_dropout_prob: float = config.hidden_dropout_prob
        self.multihead_attention: MultiHeadAttention = MultiHeadAttention(config)
        self.norm1: nn.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.norm2: nn.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.feed_forward: FeedForward = FeedForward(config)
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor, mask: torch.Tensor = None, training: bool = False) -> torch.Tensor:
        """
        Applies the encoder layer to the input hidden state.

        Args:
            hidden_state (torch.Tensor): Hidden state tensor (bs, len, dim).
            mask (torch.Tensor): Padding mask tensor (bs, len) or None.
            training (bool): Boolean flag indicating whether the layer is in training mode or not.

        Returns:
            torch.Tensor: Updated hidden state after applying the encoder layer.
        """
        x_norm1: torch.Tensor = self.norm1(hidden_state)
        attention_output: torch.Tensor = self.multihead_attention(x_norm1, x_norm1, x_norm1, mask, training)
        hidden_state: torch.Tensor = attention_output + hidden_state
        
        x_norm2: torch.Tensor = self.norm2(hidden_state)
        feed_forward_output: torch.Tensor = self.feed_forward(x_norm2)
        x_enc: torch.Tensor = feed_forward_output + hidden_state
        hidden_state: torch.Tensor = self.dropout(x_enc, training = training)
        
        return hidden_state

