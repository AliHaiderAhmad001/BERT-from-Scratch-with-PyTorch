import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from scheduled_optim import ScheduledOptim
import random as rd
import numpy as np
import argparse
from dataset import CustomTextDataset
from bert_trainer import BERTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="BERT Training Script")
    parser.add_argument("--prop", type=float, default=0.15, help="Proportion of tokens to mask in each sentence.")
    parser.add_argument("--tokenizer_path", type=str, default="bert-base-uncased", help="Path to the tokenizer.")
    parser.add_argument("--seq_len", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--delimiters", type=str, default=".,;:!?", help="Punctuation marks to split sentences.")
    parser.add_argument("--lower_case", type=bool, default=True, help="Convert text to lowercase.")
    parser.add_argument("--buffer_size", type=int, default=1, help="Number of samples to fetch and store in the buffer.")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the buffer.")
    parser.add_argument("--data_dir", type=str, default="path/to/data", help="Directory containing the data files.")
    parser.add_argument("--hidden_size", type=int, default=768, help="Size of the hidden layers.")
    parser.add_argument("--vocab_size", type=int, default=30522, help="Size of the vocabulary.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="Dropout probability for hidden layers.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_blocks", type=int, default=12, help="Number of blocks in the BERT model.")
    parser.add_argument("--final_dropout_prob", type=float, default=0.5, help="Dropout probability for the final layer.")
    parser.add_argument("--n_warmup_steps", type=int, default=10000, help="Number of warmup steps for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="Betas for the optimizer.")
    parser.add_argument("--cuda_devices", type=list, default=None, help="List of CUDA devices.")
    parser.add_argument("--with_cuda", type=bool, default=True, help="Flag to use CUDA.")
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--save_path", type=str, default='tmp/checkpoints', help="Path to save model checkpoints.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--test_dataset", type=str, default=None, help="Path to the test dataset or None.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")

    return parser.parse_args()


def set_seeds(config):
    """
    Set random seeds for reproducibility.

    Args:
        config (Config): Configuration object containing parameters.
    """
    rd.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available() and config.with_cuda:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run(config):
    """
    Main function to run BERT training.

    Args:
        config (Config): Configuration object containing parameters.
    """
    # Set random seeds
    set_seeds(config)

    print("Loading Train Dataset...")

    # Load training dataset
    train_dataset = CustomTextDataset(config)

    # Load test dataset if provided
    test_dataset = CustomTextDataset(config) if config.test_dataset is not None else None

    # Initialize BERT model
    bert = BERT(config)

    # Create data loaders
    train_data_loader = iter(DataLoader(train_dataset, batch_size=None, worker_init_fn=np.random.seed(config.seed)))
    test_data_loader = iter(DataLoader(test_dataset, batch_size=None, worker_init_fn=np.random.seed(config.seed))) if test_dataset is not None else None

    # Initialize optimizer and scheduler
    optim = Adam(bert.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    optim_schedule = ScheduledOptim(config, optim)

    # Initialize BERT trainer
    trainer = BERTTrainer(config, bert, optim_schedule, train_data_loader, test_data_loader)

    # Training loop
    for epoch in range(config.epochs):
        # Train the model
        trainer.train(epoch)

        # Save the model
        trainer.save(epoch)

        # Test the model if test data is available
        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == "__main__":
    # Load configuration
    args = parse_args()
    config = Config(
        prop=args.prop,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        delimiters=args.delimiters,
        lower_case=args.lower_case,
        buffer_size=args.buffer_size,
        shuffle=args.shuffle,
        data_dir=args.data_dir,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        final_dropout_prob=args.final_dropout_prob,
        n_warmup_steps=args.n_warmup_steps,
        weight_decay=args.weight_decay,
        lr=args.lr,
        betas=args.betas,
        cuda_devices=args.cuda_devices,
        with_cuda=args.with_cuda,
        log_freq=args.log_freq,
        batch_size=args.batch_size,
        save_path=args.save_path,
        seed=args.seed,
        test_dataset=args.test_dataset,
        epochs=args.epochs
    )


    # Run BERT training
    run(config)
