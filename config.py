class Config:
    """
    Configuration class for your model and training.

    Args:
        prop (float): Proportion of tokens to mask in each sentence. Default is 0.15.
        tokenizer_path (str): Path to the tokenizer. Default is "bert-base-uncased".
        seq_len (int): Maximum sequence length. Default is 128.
        delimiters (str): Punctuation marks to split sentences. Default is ".,;:!?".
        lower_case (bool): Flag to indicate whether to convert text to lowercase. Default is True.
        buffer_size (int): Number of samples to fetch and store in the buffer. Default is 1.
        shuffle (bool): Flag to indicate whether to shuffle the buffer. Default is True.
        data_dir (str): Directory containing the data files. Default is "path/to/data".

        # Embeddings params
        hidden_size (int): Size of the hidden layers. Default is 768.
        vocab_size (int): Size of the vocabulary. Default is 30522.
        hidden_dropout_prob (float): Dropout probability for hidden layers. Default is 0.1.

        # Attention params
        num_heads (int): Number of attention heads. Default is 8.

        # BERT model params
        num_blocks (int): Number of blocks in the BERT model. Default is 12.
        final_dropout_prob (float): Dropout probability for the final layer. Default is 0.5.

        # Optimizer params
        n_warmup_steps (int): Number of warmup steps for the optimizer. Default is 10000.
        weight_decay (float): Weight decay for the optimizer. Default is 0.01.
        lr (float): Learning rate for the optimizer. Default is 1e-4.
        betas (tuple): Betas for the optimizer. Default is (0.9, 0.999).

        # Trainer params
        cuda_devices (list): List of CUDA devices. Default is None.
        with_cuda (bool): Flag to use CUDA. Default is True.
        log_freq (int): Logging frequency. Default is 10.
        batch_size (int): Batch size for training. Default is 64.
        save_path (str): Path to save model checkpoints. Default is 'tmp/checkpoints'.

        # Run the model params
        seed (int): Random seed for reproducibility. Default is 0.
        test_dataset (str): Path to the test dataset or None. Default is None.
        epochs (int): Number of training epochs. Default is 1.
    """

    def __init__(self, prop=0.15, tokenizer_path="bert-base-uncased", seq_len=128, delimiters=".,;:!?",
                 lower_case=True, buffer_size=1, shuffle=True, data_dir="path/to/data", hidden_size=768,
                 vocab_size=30522, hidden_dropout_prob=0.1, num_heads=8, num_blocks=12, final_dropout_prob=0.5,
                n_warmup_steps=10000, weight_decay=0.01, lr=1e-4, betas=(0.9, 0.999),
                 cuda_devices=None, with_cuda=True, log_freq=10, batch_size=64, save_path='tmp/checkpoints',
                 seed=0, test_dataset=None, epochs=1):

        # Dataset params
        self.prop = prop
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.delimiters = delimiters
        self.lower_case = lower_case
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.data_dir = data_dir

        # Embeddings params
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob

        # Attention params
        self.num_heads = num_heads

        # BERT model params
        self.num_blocks = num_blocks
        self.final_dropout_prob = final_dropout_prob

        # Optimizer params
        self.n_warmup_steps = n_warmup_steps
        self.weight_decay = weight_decay
        self.lr = lr
        self.betas = betas

        # Trainer params
        self.cuda_devices = cuda_devices
        self.with_cuda = with_cuda
        self.log_freq = log_freq
        self.batch_size = batch_size
        self.save_path = save_path

        # Run the model params
        self.seed = seed
        self.test_dataset = test_dataset
        self.epochs = epochs

