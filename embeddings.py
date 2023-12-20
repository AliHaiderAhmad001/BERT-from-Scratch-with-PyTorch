import torch
import torch.nn as nn
from positional_embeddings import PositionalEmbeddings

class Embeddings(nn.Module):
    """
    Embeddings layer.

    This layer combines token embeddings with positional embeddings and segment embeddings
    to create the final embeddings.

    Args:
        config (object): Configuration object containing parameters.
            - hidden_size (int): Size of the hidden embeddings.
            - vocab_size (int): Size of the vocabulary.
            - hidden_dropout_prob (float): Dropout probability for regularization.

    Attributes:
        token_embeddings (nn.Embedding): Token embedding layer.
        positional_embeddings (PositionalEmbeddings): Positional Embeddings layer.
        segment_embeddings (nn.Embedding): Segment embedding layer.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, config):
        """
        Initializes the Embeddings layer.

        Args:
            config (object): Configuration object containing parameters.
                - hidden_size (int): Size of the hidden embeddings.
                - vocab_size (int): Size of the vocabulary.
                - hidden_dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()

        self.hidden_size: int = config.hidden_size
        self.vocab_size: int = config.vocab_size
        self.hidden_dropout_prob: float = config.hidden_dropout_prob

        self.token_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size
        )
        self.segment_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=3, embedding_dim=self.hidden_size
        )
        self.positional_embeddings: PositionalEmbeddings = PositionalEmbeddings(config)
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Forward pass of the Embeddings layer.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            segment_ids (torch.Tensor): Input tensor containing segment IDs.
            training (bool): Whether the model is in training mode.

        Returns:
            torch.Tensor: Final embeddings tensor.
        """
        pos_info: torch.Tensor = self.positional_embeddings(input_ids)
        seg_info: torch.Tensor = self.segment_embeddings(segment_ids)
        x: torch.Tensor = self.token_embeddings(input_ids)
        x: torch.Tensor = x + pos_info + seg_info
        if training:
            x: torch.Tensor = self.dropout(x)
        return x

    def forward_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the mask for the inputs.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.

        Returns:
            torch.Tensor: Computed mask tensor.
        """
        return input_ids != 0

