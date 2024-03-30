import torch
import torch.nn as nn
from encoder import Encoder
from embeddings import Embeddings

class BERT(nn.Module):
    """
    BERT model.

    Args:
        config (object): Configuration object containing hyperparameters.
            - num_blocks (int): Number of encoder blocks.
            - vocab_size (int): Size of the vocabulary.
            - d_model (int): Dimensionality of the model's hidden layers.
            - hidden_size (int): Size of the hidden embeddings.

    Attributes:
        num_blocks (int): Number of encoder blocks.
        vocab_size (int): Size of the vocabulary.
        final_dropout_prob (float): Dropout probability for the final layer.
        hidden_size (int): Size of the hidden embeddings.
        embed_layer (Embeddings): Embeddings layer.
        encoder (nn.ModuleList): List of encoder layers.
        mlm_prediction_layer (nn.Linear): Masked Language Model (MLM) prediction layer.
        nsp_classifier (nn.Linear): Next Sentence Prediction (NSP) classifier layer.
        softmax (nn.LogSoftmax): LogSoftmax layer for probability computation.
    """

    def __init__(self, config):
        """
        Initializes the BERT model.
        """
        super(BERT, self).__init__()

        self.num_blocks: int = config.num_blocks
        self.vocab_size: int = config.vocab_size
        self.hidden_size: int = config.hidden_size

        self.embed_layer: Embeddings = Embeddings(config)
        self.encoder: nn.ModuleList = nn.ModuleList([Encoder(config) for _ in range(self.num_blocks)])
        self.mlm_prediction_layer: nn.Linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.nsp_classifier: nn.Linear = nn.Linear(self.hidden_size, 2)
        self.softmax: nn.LogSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the BERT model.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            segment_ids (torch.Tensor): Input tensor containing segment IDs.
            training (bool): Whether the model is in training mode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: MLM outputs and NSP outputs.
        """
        x_enc: torch.Tensor = self.embed_layer(input_ids, segment_ids, training)
        mask = self.embed_layer.forward_mask(input_ids)

        for encoder_layer in self.encoder:
            x_enc: torch.Tensor = encoder_layer(x_enc, mask, training=training)

        mlm_logits: torch.Tensor = self.mlm_prediction_layer(x_enc)
        nsp_logits: torch.Tensor = self.nsp_classifier(x_enc[:, 0, :])

        return self.softmax(mlm_logits), self.softmax(nsp_logits)

