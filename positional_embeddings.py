import torch
import torch.nn as nn

class PositionalEmbeddings(nn.Module):
    """
    PositionalEmbeddings layer.

    This layer generates positional embeddings based on input IDs.
    It uses an Embedding layer to map position IDs to position embeddings.

    Args:
        config (object): Configuration object containing parameters.
            - seq_len (int): Maximum sequence length.
            - hidden_size (int): Size of the hidden embeddings.
    """

    def __init__(self, config):
        """
        Initializes the PositionalEmbeddings layer.

        Args:
            config (object): Configuration object containing parameters.
                - seq_len (int): Maximum sequence length.
                - hidden_size (int): Size of the hidden embeddings.
        """
        super().__init__()

        self.seq_len: int = config.seq_len
        self.hidden_size: int = config.hidden_size
        self.positional_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.seq_len, embedding_dim=self.hidden_size
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate positional embeddings.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.

        Returns:
            torch.Tensor: Positional embeddings tensor of shape (batch_size, seq_length, hidden_size).
        """
        seq_length: int = input_ids.size(1)
        position_ids: torch.Tensor = torch.arange(seq_length, dtype=torch.int32, device=input_ids.device).unsqueeze(0)
        position_embeddings: torch.Tensor = self.positional_embeddings(position_ids)
        return position_embeddings

