import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Feed-forward layer implementation.

    Args:
        config (object): Configuration object containing hyperparameters.
            - hidden_size (int): Hidden size for the model (embedding dimension).
            - hidden_dropout_prob (float): Dropout probability for regularization.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        intermediate_fc_size (int): Intermediate size for the fully connected layers.
        hidden_dropout_prob (float): Dropout probability for regularization.
        fc1 (nn.Linear): First linear layer.
        fc2 (nn.Linear): Second linear layer.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, config):
        """
        Initializes the FeedForward layer.

        Args:
            config (object): Configuration object containing hyperparameters.
                - hidden_size (int): Hidden size for the model (embedding dimension).
                - hidden_dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()

        self.hidden_size: int = config.hidden_size
        self.intermediate_fc_size: int = self.hidden_size * 4
        self.hidden_dropout_prob: float = config.hidden_dropout_prob

        self.fc1: nn.Linear = nn.Linear(self.hidden_size, self.intermediate_fc_size)
        self.fc2: nn.Linear = nn.Linear(self.intermediate_fc_size, self.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Applies feed-forward transformation to the input hidden state.

        Args:
            hidden_state (torch.Tensor): Hidden state tensor (batch_size, sequence_length, hidden_size).
            training (bool): Boolean indicating whether the model is in training mode or inference mode.

        Returns:
            torch.Tensor: Updated hidden state after applying feed-forward transformation.
        """
        hidden_state: torch.Tensor = self.fc1(hidden_state)
        hidden_state: torch.Tensor = F.gelu(hidden_state)
        hidden_state: torch.Tensor = self.dropout(hidden_state, training=training)
        hidden_state: torch.Tensor = self.fc2(hidden_state)
        
        return hidden_state

