# BERT-from-Scratch-with-PyTorch/demo

## Download Dataset

We'll be dealing with a relatively small dataset, the popular Imdb movie dataset (you can use a larger dataset as large as you want, but this will require additional resources and perhaps some modifications to the Dataset class).
The entire code was tested and built on Colab, but also made it possible to run it locally via Commandline.

```
%cd '/content/drive/MyDrive/Colab Notebooks/projects/BERTvolution-Building-BERT-from-Scratch/datasets'
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

## Project directory
I assume that you have a folder called projects and inside it is a project folder called BERT-from-Scratch-with-PyTorch.
```
%cd '/content/drive/MyDrive/Colab Notebooks/projects/BERTvolution-Building-BERT-from-Scratch'
```

## Download requirements
```!pip install transformers```

## Overview

**Note:** Be careful, I may not be doing the same order of steps or even exactly the same approach. This is just a principled approach I put together to make the steps clearer in principle. I wrote this pseudocode before doing anything, and did not modify it later.

---

### **Generate_samples**

I have written the following sudo code that more or less expresses the sequence of operations to generate samples of each document from the dataset:
```
Generate_samples_from_doc(doc):
1. samples=[]
2. doc=preprocessing(doc)
3. sentences=doc.split('.')
4. for current_sent in sentences:
       4.1. With_a_50%_probability() And There_Is_a_next_sentence():
                 4.1.1. next_sentence= read_the_following_sentence()
                 4.1.2. is_next=True
       4.2. else:
                 4.2.1. next_sentence = read_random_sentence_from_another_doc()
                 4.2.2. is_next=False

       4.3. S1, S2 = Masking(current_sent, next_sentence)
       4.4. SEQ = add_special_tokens(S1, S2)
       4.5. SEQ = padding(SEQ, MAX_LEN)
       4.6. SEG = segmentation(S1_LEN, S2_LEN, MAX_LEN)
       4.11. samples.add([SEQ, SEG, is_next])
5. return samples
```

Training a BERT-based model involves creating a dataset of training samples, each comprising a pair of sentences with appropriate masking, padding, and segmentation.

1. We start by creating an empty list that will hold our training samples.
2. Ensure the document undergoes any necessary preprocessing steps before generating samples. This may include tasks like removing special characters or irrelevant information.
3. Divide the document into individual sentences. This step assumes that the sentences are terminated by periods, and further adjustments may be needed based on the nature of the text.
4. Loop through each sentence in the document:

  4.1. Decide Whether to Create a "Next Sentence" Pair: With a 50% probability, decide whether to create a "next sentence" pair. If yes, read the following sentence from the current document; otherwise, select a random sentence from another document. Set is_next accordingly.

  4.3. Mask and Combine Sentences: Apply masking to the current sentence and the selected next sentence. Combine them, adding special tokens as required by the BERT model.

  4.5. Pad the Sequence: Ensure that the combined sequence adheres to the maximum length (MAX_LEN) required by the model by padding if necessary.

  4.6. Segment the Sequence: Create a segmentation mask based on the lengths of the masked sentences and the maximum sequence length.

  4.11. Add the Sample to the List: Combine the sequence, segmentation mask, and the label indicating whether the pair represents consecutive sentences (is_next). Add this sample to the list.

5. Return the Generated Samples.


#### **Objective Training in BERT**

Central to BERT's success is its unique training objective. At the heart of BERT's training is the Masked Language Model (MLM) objective. During pre-training, random words in a sentence are masked, and the model is tasked with predicting these masked words. This bidirectional approach, where both left and right contexts contribute to the prediction, allows BERT to capture contextual information effectively. More precisely, MLM includes three points:
* Replace the word with a mask [MASK]: Here the model must find out the hidden word. 
Example:
  
      Input Sequence  : The man went to [MASK] store with [MASK] dog
      Target Sequence :                  the                his

* Replace the word with a random word. Here the model must detect that this word is not relevant, so it predicts the original word.
* Do not replace the word with anything. Here the model must know that this word is appropriate and in its correct place.

BERT also incorporates the Next Sentence Prediction (NSP) objective during pre-training. In this task, pairs of sentences are provided to the model, and it learns to predict whether the second sentence follows the first in the original document. This encourages the model to understand relationships between sentences and grasp document-level context. The combination of MLM and NSP objectives is pivotal. While MLM focuses on capturing word-level semantics, NSP contributes to understanding the coherence and flow of sentences within a document. This dual-objective pre-training equips BERT with a robust understanding of both local and global context in language.

BERT's objective training follows a transfer learning paradigm. Pre-training on a massive corpus allows the model to learn general language representations. Fine-tuning on downstream tasks, such as sentiment analysis or named entity recognition, then refines these representations for specific applications. This approach has proven highly effective in achieving state-of-the-art results across various NLP tasks.

#### **Samples**
Each training data sample should be in the following format:

`SentA(str), SentB(str), Labels(List[int]), is_next(int)`

`SentA`: The first sentence.
`SentB`: The second sentence.

The second sentence can be the sentence immediately after the first sentence in the document, or it can be a sentence read from another document at random.

`Labels`: Each word in each data sample will be either masked (we replace it with `["MASK"]`), replaced (we replace it with a random word), or nothing (we leave it as is). In the first and second cases, the label is the original word (or its idx), and in the third case, the label is `0`.

`is_next`: To indicate whether the two sentences are consecutive or not.

#### **Splitting text**

I have seen many implementations of the BERT architecture on GitHub, and unfortunately I've seen many misconceptions resulting from a wrong expectation or understanding of the original paper. For example, all implementations I've seen break sentences based on a period (`.`).

In natural language processing tasks, splitting text into sentences is not always straightforward and can vary based on the specific requirements of the task at hand. While splitting by periods (`.`) is a common practice, it might not capture all valid sentence boundaries, especially in contexts where sentences are not neatly separated by periods. The matter is more complicated than that.

According to BERT paper:
> "Throughout this work, a “sentence” can be an arbitrary span of contiguous text, rather than an actual linguistic sentence."


This statement clarifies the definition of a "sentence" in the context of BERT. Traditionally, in linguistics, a sentence is defined as a grammatically complete sequence of words expressing a thought, question, command, or exclamation. However, in the context of BERT and many natural language processing (NLP) tasks, the definition is more flexible. In BERT, for example, a "sentence" is not constrained by traditional grammatical rules. Instead, it refers to any contiguous span of text in the input data. This span can encompass multiple phrases, clauses, or even a full paragraph, depending on the task requirements. This flexibility allows BERT to handle a wide range of tasks where sentence boundaries might not align with traditional linguistic sentences. For instance, in document-level sentiment analysis, each document might be treated as a single "sentence" for analysis, even if it contains multiple paragraphs.


#### **Special Tokens**

The first sentence must begin with the symbol [CLS], the second sentence must end with the symbol [SEP], and the two sentences must be separated by the symbol [SEP]. Example:

      Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
      Is Next : Yes

      Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
      Is Next = No


#### **Segmentation**

To differentiate between segments, BERT uses segment embeddings. Each token is assigned a segment embedding indicating to which segment it belongs. For example, in a sentence pair, tokens from the first sentence might have a segment embedding of 0, and tokens from the second sentence might have a segment embedding of 1.

      Input:      "[CLS] I love natural language processing . [SEP] It is fascinating . [SEP]"

      SegmentIDS:  1  1  1  1  1  1  1  1  0  0  0  0  0  0


## Configration

```
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

```

## Tokenizers

BERT typically uses a subword tokenizer known as WordPiece or SentencePiece. These subword tokenization techniques are specifically designed to handle the complexity of various languages and efficiently tokenize text into smaller units, making them suitable for BERT and other Transformer-based models. Let's take a closer look at each of these tokenization methods:

1. **WordPiece Tokenization:**

   - **Description:** WordPiece tokenization is a subword tokenization method that breaks down words into smaller units, such as subword pieces or characters. It starts with a predefined vocabulary of subword pieces and aims to represent most words as combinations of these pieces.
   
   - **Vocabulary:** The vocabulary for WordPiece tokenization is typically built based on the frequency of subword pieces in a large corpus of text. It includes commonly occurring subword pieces and special tokens like [CLS], [SEP], and [MASK].

   - **Example:** The word "unhappiness" might be tokenized into subword pieces like ["un", "##h", "##ap", "##py", "##ness"].

   - **Usage in BERT:** BERT often uses WordPiece tokenization, and its pretraining is done with WordPiece tokenized text.
   - **Click [here](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt) for more.**

2. **SentencePiece Tokenization:**

   - **Description:** SentencePiece is another subword tokenization technique that is more data-driven and flexible than WordPiece. It allows for tokenization at the subword level and is capable of handling various languages and scripts effectively.

   - **Vocabulary:** Unlike WordPiece, SentencePiece does not rely on a predefined vocabulary. Instead, it uses unsupervised training to learn a subword vocabulary directly from the input text.

   - **Example:** SentencePiece can tokenize the word "unhappiness" into subword pieces like ["▁un", "happiness"].

   - **Usage in BERT:** SentencePiece is also commonly used with BERT, especially when dealing with multilingual or less common languages where predefined vocabularies may not be readily available.
   - **Click [here](https://huggingface.co/docs/transformers/tokenizer_summary#sentencepiece) for more.**

* **When do we use either?** The choice between WordPiece and SentencePiece often depends on the specific use case, language requirements, and the availability of pretrained models and tokenizers.

* #### Adapting WordPiece tokenizer

* The Hugging Face Transformers library provides a unified interface for various models, including BERT. When you use the AutoTokenizer class from the library, it loads the appropriate tokenizer based on the model name you provide.

We can build the tokenizer from scratch or we can use a ready-made implementation of it. I don't like the idea of rebuilding it at all, it's a tedious and complex process, and it seems off topic. Common practice is to reuse and adapt the same implementation on a new dataset or corpus.

It should be noted that tokenizer training continues until a certain number of iterations is reached, a certain number of vocabulary are reached, convergence is reached, or until the algorithm finds no new pairs to combine (in the latter case we get all the unique vocabulary in the corpus). That is, there are 4 different ways the training can be stopped. In this repo, we will use the most popular method, the second method (when training from scratch, I think the third method is the best, but it is more tiring than the other methods).

We already know that the BERT model has a vocab size of 30522 I think, so I can stop the algorithm when I get to 30622. I think that will suffice and more since we are dealing with a relatively small and familiar data set. Actually, we never need to retrain the tokenizer for this dataset, but it's nice to get our hands dirty sometimes. However, you can still skip this step and use the tokenizer without training it.

```
import os
import concurrent.futures
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from typing import List, Iterator
import argparse

class TokenizerTrainer(ABC):
    """
    Abstract base class for training tokenizers.

    Args:
        tokenizer_name (str): The name of the model that the tokenizer is associated with in order to load it.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): Pretrained tokenizer.

    Methods:
        train(data_dir, batch_size, vocab_size, save, save_fp): Train the tokenizer on a new corpus.
        read_batch_of_files(data_dir, batch_size): Read a batch of files from a directory.
        save(fp): Save the adapted tokenizer to a file.
        get_filenames(data_dir): Get a list of filenames in a directory.
        read_file(filename): Read the contents of a file.

    """
    def __init__(self, tokenizer_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def train(self, data_dir: str, batch_size: int, vocab_size: int, save_fp: str) -> None:
        """
        Train the tokenizer on a new corpus.

        Args:
            data_dir (str): Corpus directory path.
            batch_size (int): Batch size for reading files.
            vocab_size (int): Target vocabulary size.
            save_fp (str): File path to save the tokenizer.

        """
        training_corpus = self.read_batch_of_files(data_dir, batch_size)
        self.tokenizer = self.tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        self.save(save_fp)

    def read_batch_of_files(self, data_dir: str, batch_size: int) -> Iterator[List[str]]:
        """
        Read a batch of files from a directory.

        Args:
            data_dir (str): Directory path.
            batch_size (int): Batch size.

        Yields:
            list: A batch of file contents as a list of strings.

        """
        filenames = self.get_filenames(data_dir)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for start_idx in range(0, len(filenames), batch_size):
                batch_filenames = filenames[start_idx : start_idx + batch_size]
                batch_contents = []
                future_to_filename = {executor.submit(self.read_file, filename): filename for filename in batch_filenames}
                for future in concurrent.futures.as_completed(future_to_filename):
                    filename = future_to_filename[future]
                    content = future.result()
                    batch_contents.append(content)

                yield batch_contents

    def save(self, fp: str) -> None:
        """
        Save the adapted tokenizer to a file.

        Args:
            fp (str): File path to save the tokenizer.

        """
        self.tokenizer.save_pretrained(fp)

    @abstractmethod
    def get_filenames(self, data_dir: str) -> List[str]:
        """
        Abstract method to get a list of filenames in a directory.

        Args:
            data_dir (str): Directory path.

        Returns:
            list: List of filenames.

        """
        pass

    @abstractmethod
    def read_file(self, filename: str) -> str:
        """
        Abstract method to read the contents of a file.

        Args:
            filename (str): File path.

        Returns:
            str: Content of the file as a string.

        """
        pass

class WordPieceTrainer(TokenizerTrainer):
    """
    Tokenizer trainer for training a WordPiece tokenizer.

    Inherits from TokenizerTrainer.

    Methods:
        get_filenames(data_dir): Get a list of filenames in a directory.
        read_file(filename): Read the contents of a file.

    """
    def get_filenames(self, data_dir: str) -> List[str]:
        """
        Retrieves a list of filenames from the 'neg' and 'pos' directories within the 'train' and 'test' directories.

        Args:
            data_dir (str): The top-level directory containing the data files.

        Returns:
            list: List of file paths.
        """
        filenames = []

        for root, dirs, files in os.walk(data_dir):
            if os.path.basename(root) in ['neg', 'pos'] and os.path.basename(os.path.dirname(root)) in ['train', 'test']:
                for file in files:
                    filepath = os.path.join(root, file)
                    filenames.append(filepath)
        return filenames

    def read_file(self, filename: str) -> str:
        """
        Read the contents of a file.

        Args:
            filename (str): File path.

        Returns:
            str: Content of the file as a string.

        """
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        return content

def main():
  tokenizer_name = "bert-base-cased"
  data_dir = "aclImdb"
  batch_size = 1000
  vocab_size = 30522
  save_fp = "tokenizer/adapted-tokenizer"
  wp_trainer = WordPieceTrainer(tokenizer_name=tokenizer_name)
  wp_trainer.train(data_dir=data_dir, batch_size=batch_size, vocab_size=vocab_size, save_fp=save_fp)

if __name__ == "__main__":
    main()
```

#### Tokenizer

In fact, you can rebuild the tokenizer from scratch (it's not a difficult issue), but I preferred to use the transformer library in order to skip building it from scratch and training it.

```
# tokenizer.py
from transformers import BertTokenizerFast, logging
from abc import ABCMeta, abstractmethod
from typing import List, Dict

# Set the logging level for the transformers library to ERROR
logging.set_verbosity(logging.ERROR)

class DataTokenizer(metaclass=ABCMeta):
    """
    Abstract base class for tokenizing and detokenizing sentences.

    Attributes:
        None

    Methods:
        convert_to_tokens(sentence: str) -> List[str]:
            Tokenizes a sentence and converts it to tokens.

        convert_to_ids(tokens: List[str]) -> List[int]:
            Converts a list of tokens to a list of token IDs.

        detokenize(token_ids: List[int]) -> str:
            Converts token IDs back to a sentence.
    """

    @abstractmethod
    def convert_to_tokens(self, sentence: str) -> List[str]:
        """
        Tokenizes a sentence and converts it to tokens.

        Args:
            sentence (str): The input sentence to be tokenized.

        Returns:
            List[str]: List of tokens.
        """

    @abstractmethod
    def convert_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Converts a list of tokens to a list of token IDs.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[int]: List of token IDs.
        """

    @abstractmethod
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Converts token IDs back to a sentence.

        Args:
            token_ids (List[int]): List of token IDs.

        Returns:
            str: Detokenized sentence.
        """


class EnglishDataTokenizer(DataTokenizer):
    """
    Tokenizer class for English sentences using the Hugging Face transformers library.

    Attributes:
        tokenizer (BertTokenizerFast): Hugging Face BERT tokenizer instance.
        max_length (int): Maximum sequence length.

    Methods:
        get_vocab() -> Dict[str, int]:
            Returns the vocabulary of the tokenizer.

        token_to_id(token: str) -> int:
            Converts a token to its corresponding ID.

        convert_to_tokens(sentence: str) -> List[str]:
            Converts a sentence to a list of tokens.

        convert_to_ids(tokens: List[str]) -> List[int]:
            Converts a list of tokens to a list of token IDs.

        detokenize(token_ids: List[int]) -> str:
            Converts a list of token IDs back to a sentence.
    """

    def __init__(self, tokenizer_path: str, max_length: int):
        """
        Initializes the EnglishDataTokenizer.

        Args:
            tokenizer_path (str): Path to the pre-trained tokenizer.
            max_length (int): Maximum sequence length.
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        self.max_length = max_length

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary of the tokenizer.

        Returns:
            Dict[str, int]: Vocabulary of the tokenizer.
        """
        return self.tokenizer.get_vocab()

    def token_to_id(self, token: str) -> int:
        """
        Converts a token to its corresponding ID.

        Args:
            token (str): Input token.

        Returns:
            int: Corresponding token ID.
        """
        return self.tokenizer.get_vocab()[token]

    def convert_to_tokens(self, sentence: str) -> List[str]:
        """Tokenizes a sentence and converts it to tokens."""
        return self.tokenizer.tokenize(sentence)

    def convert_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts a list of tokens to a list of token IDs."""
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def detokenize(self, token_ids: List[int]) -> str:
        """Converts token IDs back to a sentence."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
```

## Data Loader


**Note:** In fact, I do not think that this method is ideal. The best thing is to create new documents, so that each line in each document contains only one sample, but for some reason I decided to deal with the data as it is (raw). In the current way I'm using it, we read a document and then have the `__getitem__` function generate all possible samples from it. The number of samples that can be generated from each document is unknown (maybe one sample, maybe more), and therefore we are forced to disable the batching process when wrapping the dataset in `DataLoader`, and then form the batches manually within the training loop (whenever we read 32 samples we pass them to the model, and the surplus we use them in the next iteration, and so on). So the best approach is to do as I mentioned at the beginning of the note, so that we do not have to disable batching.

---


**CustomTextDataset for BERT-style Pre-training**

The provided code defines a `CustomTextDataset` class, which serves as a PyTorch dataset designed for pre-training BERT-like models on text data. The dataset is configured using a `Config` object containing various parameters.

**Attributes:**

- **MASK_TOKEN, PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, PAD_IDX:** Special tokens used in BERT-like models for masking, padding, starting a sequence, and separating segments.

- **prop:** Proportion of tokens to mask in each sentence.

- **tokenizer_path, seq_len, shuffle:** Configuration parameters for the tokenizer path, maximum sequence length, and whether to shuffle the dataset.

- **buffer_idx, buffer, ptr:** Variables for managing the buffer, a list storing samples, and a pointer for fetching from the buffer.

- **max_words:** Maximum number of words in one sentence.

- **delimiters, lower_case, buffer_size, data_dir:** Parameters related to sentence splitting, case conversion, buffer size, and the directory containing data files.

- **filenames:** List of file paths representing the data documents.

- **tokenizer:** An instance of the `EnglishDataTokenizer` class for tokenizing sentences.

- **bert_vocab:** Vocabulary list for the BERT model, excluding unused tokens and special tokens.

**Methods:**

- **__init__(self, config: Config):** Initializes the dataset with configuration parameters.

- **__len__(self) -> int:** Returns the total number of data documents in the dataset.

- **__getitem__(self, index: int) -> Dict[str, List[int]]:** Retrieves an item (sample) from the dataset.

- **fetch_to_buffer(self) -> None:** Fetches samples to the buffer for the next iteration.

- **generate_samples(self, doc_name: str) -> List[Dict[str, List[int]]]:** Generates samples from a document.

- **split_sentences(self, text: str, delimiters: str, max_words: int) -> List[str]:** Splits text into sentences based on various strategies.

- **mask_sentence(self, sent: str) -> Tuple[List[str], List[str]]:** Masks tokens in a sentence.

- **read_random_sentence(self) -> str:** Reads a random sentence from the dataset.

- **get_filenames(self, data_dir: str) -> List[str]:** Retrieves a list of filenames from specific directories.

- **reset(self) -> None:** Resets the buffer index and refills the buffer for the next iteration.

**Description:**

The dataset is designed for BERT-style pre-training, a two-segment model that learns contextualized word representations. It uses a buffer mechanism to efficiently manage large datasets, and the `generate_samples` method creates training samples by masking tokens in sentences. The dataset supports various configuration options for tokenization, sequence length, shuffling, and more. It reads data from specified directories, tokenizes it using the provided tokenizer, and generates samples suitable for pre-training BERT-like models. This code is a fundamental component for building a PyTorch dataset tailored for pre-training transformer models on text data.


**Note: Buffer Size and Efficiency.**

  The buffer size you choose can significantly impact the efficiency of your data loading. A smaller buffer size means more frequent disk reads but lower memory usage, while a larger buffer size reduces disk reads but increases memory consumption. You should choose a buffer size that strikes a balance between these trade-offs based on your available resources and performance requirements.

```
import os
import re
from typing import List, Tuple, Dict
import random as rd
from math import ceil
from torch.utils.data import DataLoader, Dataset


class CustomTextDataset(Dataset):
    """
    Custom PyTorch Dataset for BERT-style pre-training on text data.

    Args:
        config (Config): Configuration object with dataset parameters.

    Attributes:
        MASK_TOKEN (str): Special token representing a masked word.
        PAD_TOKEN (str): Special token for padding sequences.
        CLS_TOKEN (str): Special token marking the start of a sequence.
        SEP_TOKEN (str): Special token marking the separation of segments.
        PAD_IDX (int): Index of the padding token in the vocabulary.
        prop (float): Proportion of tokens to mask in each sentence.
        tokenizer_path (str): Path to the pre-trained tokenizer.
        seq_len (int): Maximum sequence length.
        shuffle (bool): Flag indicating whether to shuffle the dataset.
        buffer_idx (int): Index for the current position in the buffer.
        buffer (List[Dict[str, List[int]]]): Buffer for storing samples.
        ptr (int): Pointer for the current position when fetching from the buffer.
        max_words (int): Maximum number of words in one sentence.
        delimiters (str): Punctuation marks for sentence splitting.
        lower_case (bool): Flag indicating whether to convert sentences to lowercase.
        buffer_size (int): Size of the buffer for fetching samples.
        data_dir (str): Directory containing data files.
        filenames (List[str]): List of file paths.
        tokenizer (EnglishDataTokenizer): Instance of the EnglishDataTokenizer.
        bert_vocab (List[str]): Vocabulary list for BERT model.
    """
    MASK_TOKEN = '[MASK]'
    PAD_TOKEN = '[PAD]'
    CLS_TOKEN = '[CLS]'
    SEP_TOKEN = '[SEP]'
    PAD_IDX = 0

    def __init__(self, config: Config):
        """
        Initialize the CustomTextDataset.

        Args:
            config (Config): Configuration object with dataset parameters.
        """

        self.prop = config.prop
        self.tokenizer_path = config.tokenizer_path
        self.seq_len = config.seq_len
        self.shuffle = config.shuffle
        assert self.seq_len >= 8

        self.buffer_idx = 0
        self.buffer: List[Dict[str, List[int]]] = []
        self.ptr = 0
        self.max_words = self.seq_len // 2  # Maximum number of words in one sentence

        self.delimiters = config.delimiters
        self.lower_case = config.lower_case
        self.buffer_size = config.buffer_size
        self.data_dir = config.data_dir
        self.filenames = self.get_filenames(self.data_dir)
        self.tokenizer = EnglishDataTokenizer(self.tokenizer_path, self.seq_len)
        self.fetch_to_buffer()
        vocab = self.tokenizer.get_vocab()
        self.bert_vocab = [
            word for word in vocab.keys()
            if not (re.compile(r'\[unused\d+\]').match(word)
                    or word in ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
                    or not re.compile(r'^[a-zA-Z]+$').match(word))
        ]

    def __len__(self) -> int:
        """Returns the total number of data documents in the dataset."""
        return len(self.filenames)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            dict: Dictionary containing dataset samples.
        """
        idx = index % self.buffer_size

        if (index + 1) == len(self.filenames):
            x = self.buffer[idx]
            self.reset()
            return x

        if ((idx + 1) % self.buffer_size) == 0:
            self.fetch_to_buffer()

        return self.buffer[idx]

    def fetch_to_buffer(self) -> None:
        """Fetches samples to the buffer for the next iteration."""
        self.buffer = self.buffer[self.ptr:]
        self.ptr = 0

        samples = [self.generate_samples(filename) for filename in
                   self.filenames[self.buffer_idx:self.buffer_idx + self.buffer_size]]
        self.buffer.extend(samples)
        self.buffer_idx += self.buffer_size

        if self.shuffle:
            rd.shuffle(self.buffer)

    def generate_samples(self, doc_name: str) -> List[Dict[str, List[int]]]:
        """
        Generates samples from a document.

        Args:
            doc_name (str): Name of the document.

        Returns:
            List[Dict[str, List[int]]]: List of samples.
        """
        def custom_std(sentence: str, lower_case: bool = False) -> str:
            """ Remove HTML line-break tags and lowercase the sentence."""
            sentence = re.sub("<br />", " ", sentence).strip()
            if self.lower_case:
                sentence = sentence.lower()
            return sentence

        with open(doc_name, "r") as file:
            doc = file.read().strip()

        if doc == '':
            return []

        samples = []
        sentences = self.split_sentences(doc, self.delimiters, self.max_words)

        last_sentence = len(sentences) - 1
        for i, sent_A in enumerate(sentences):
            if rd.random() <= 0.5 and i != last_sentence:
                is_next = 1
                sent_B = sentences[i + 1]
            else:
                is_next = 0
                sent_B = self.read_random_sentence()

            sent_A, sent_B = custom_std(sent_A, self.lower_case), custom_std(sent_B, self.lower_case)

            sent_A, label_A = self.mask_sentence(sent_A)
            sent_B, label_B = self.mask_sentence(sent_B)

            bert_label = ([CustomTextDataset.PAD_TOKEN] + label_A + [CustomTextDataset.PAD_TOKEN] + label_B) + [CustomTextDataset.PAD_TOKEN]

            sent_A = [CustomTextDataset.CLS_TOKEN] + sent_A + [CustomTextDataset.SEP_TOKEN]
            sent_B = sent_B + [CustomTextDataset.SEP_TOKEN]

            segment_label = [1 for _ in range(len(sent_A))] + [2 for _ in range(len(sent_B))]

            sequence = sent_A + sent_B

            padding = [CustomTextDataset.PAD_TOKEN for _ in range(self.seq_len - len(sequence))]
            sequence.extend(padding), bert_label.extend(padding), segment_label.extend([CustomTextDataset.PAD_IDX] * len(padding))

            bert_input = self.tokenizer.convert_to_ids(sequence)
            bert_label = self.tokenizer.convert_to_ids(bert_label)

            assert len(bert_input) == len(bert_label) == len(segment_label)

            samples.append({"bert_input": bert_input,
                            "bert_label": bert_label,
                            "segment_label": segment_label,
                            "is_next": is_next})

        return samples

    def split_sentences(self, text: str, delimiters: str, max_words: int) -> List[str]:
        """
        Splits text into sentences based on various strategies.

        Args:
            text (str): The input text to be split.
            delimiters (str, optional): Punctuation marks to split sentences.
            max_words (int, optional): The maximum number of words per split.

        Returns:
            List[str]: List of split sentences.
        """
        def split_text_by_punctuation(text: str, delimiters: str, max_words: int) -> List[str]:
            """
            Splits text into sentences based on specified punctuation marks and a maximum number of words per split.

            Args:
                text (str): The input text to be split.
                delimiters (str): Punctuation marks to split sentences.
                max_words (int): The maximum number of words per split.

            Returns:
                List[str]: List of split sentences.
            """
            sentences = []

            for sentence in re.split(r'(?<=[{}])'.format(re.escape(delimiters)), text):
                sentence = sentence.strip()
                if sentence:
                    sentences.extend(split_text_by_maximum_word_count(sentence, max_words))

            return sentences

        def split_text_by_maximum_word_count(text: str, max_words: int) -> List[str]:
            """
            Splits text into smaller strings, each with a predetermined word count.

            Args:
                text (str): The input text to be split.
                max_words (int): The maximum number of words per split.

            Returns:
                List[str]: List of split sentences.
            """
            words = text.split()
            result_sentences = []

            while words:
                result_sentences.append(" ".join(words[:max_words]))
                words = words[max_words:]

            return result_sentences

        if rd.random() < 0.75:
            return split_text_by_punctuation(text, delimiters, max_words)
        else:
            return split_text_by_maximum_word_count(text, max_words)

    def mask_sentence(self, sent: str) -> Tuple[List[str], List[str]]:
        """
        Masks tokens in a sentence.

        Args:
            sent (str): The input sentence.

        Returns:
            Tuple[List[str], List[str]]: Tuple containing masked tokens and target sequence.
        """
        tokens = self.tokenizer.convert_to_tokens(sent)[:self.max_words - 2]

        num_masked = ceil(self.prop * len(tokens))
        masked_indices = rd.sample(range(len(tokens)), num_masked)

        target_sequence = [CustomTextDataset.PAD_TOKEN] * len(tokens)

        for idx in masked_indices:
            target_sequence[idx] = tokens[idx]

            p = rd.random()

            if p < 0.8:
                tokens[idx] = CustomTextDataset.MASK_TOKEN

                next_idx = idx + 1
                while next_idx < len(tokens) and tokens[next_idx].startswith("##"):
                    tokens[next_idx] = CustomTextDataset.MASK_TOKEN
                    target_sequence[next_idx] = tokens[next_idx]
                    next_idx += 1
            elif p <= 0.5:
                tokens[idx] = rd.choice(self.bert_vocab)
            else:
                pass

        return tokens, target_sequence

    def read_random_sentence(self) -> str:
        """
        Reads a random sentence from the dataset.

        Returns:
            str: A randomly selected sentence.
        """
        idx = rd.randint(0, len(self.filenames) - 1)
        with open(self.filenames[idx], "r") as file:
            doc = file.read().strip()

        sentences = self.split_sentences(doc, self.delimiters, self.max_words)
        return sentences[rd.randint(0, len(sentences) - 1)]

    def get_filenames(self, data_dir: str) -> List[str]:
        """
        Retrieves a list of filenames from the 'neg' and 'pos' directories within the 'train' and 'test' directories.

        Args:
            data_dir (str): The top-level directory containing the data files.

        Returns:
            list: List of file paths.
        """
        filenames = []

        for root, dirs, files in os.walk(data_dir):
            if os.path.basename(root) in ['neg', 'pos'] and os.path.basename(os.path.dirname(root)) in ['train', 'test']:
                for file in files:
                    filepath = os.path.join(root, file)
                    filenames.append(filepath)
        return filenames



    def reset(self) -> None:
        """Resets the buffer index and refills the buffer for the next iteration."""
        self.buffer_idx = 0
        self.fetch_to_buffer()

```

## BERT Bulding Blocks

All the components I talked about in the [Neural-Machine-Translator](https://github.com/AliHaiderAhmad001/Neural-Machine-Translator) project, you can find there [Demo](https://github.com/AliHaiderAhmad001/Neural-Machine-Translator/tree/main/demo). Explains everything.

In general, there are some modifications from the 2017 Transformers model, which I will summarize as follows:
*We only need the Encoder. We do not use the decoder in BERT.
* We need to modify the Embeddings layer, as we are adding the Segmentation embeds.
* We also need to add a second head to the final model, for the NSP task.

### Embeddings

#### Pos_Embeddings
```
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
```

#### Embeddings layer

```
import torch
import torch.nn as nn
#from positional_embeddings import PositionalEmbeddings

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
        norm (nn.LayerNorm): Layer normalization for normalization.
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
        self.norm: nn.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)

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
        x: torch.Tensor = self.norm(x)
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
```

### Encoder

#### Attination Head

```
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

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
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
        n_value: torch.Tensor = torch.matmul(att_weights, value)

        return n_value

```

#### Multi-head attention

```
import torch
import torch.nn as nn

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

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
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
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state
```

#### The Feed-Forward Layer and Normalization

```
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
        hidden_state: torch.Tensor = self.fc2(hidden_state)
        if training:
            hidden_state: torch.Tensor = self.dropout(hidden_state)
        return hidden_state
```

#### Encoder layer

```
import torch
import torch.nn as nn
#from attention import MultiHeadAttention
#from feed_forward import FeedForward

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
        self.norm1: nn.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.norm2: nn.LayerNorm = nn.LayerNorm(self.hidden_size)
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

        attention_output: torch.Tensor = self.multihead_attention(hidden_state, hidden_state, hidden_state, mask)  # Apply multi-head attention
        hidden_state: torch.Tensor = self.norm1(attention_output + hidden_state)  # Add skip connection and normalize
        feed_forward_output: torch.Tensor = self.feed_forward(hidden_state)  # Apply feed-forward layer
        hidden_state: torch.Tensor = self.norm2(feed_forward_output + hidden_state)  # Add skip connection and normalize
        if training:
            hidden_state: torch.Tensor = self.dropout(hidden_state)
        return hidden_state
```

### BERT Model

```
import torch
import torch.nn as nn
#from encoder import Encoder
#from embeddings import Embeddings

class BERT(nn.Module):
    """
    BERT model.

    Args:
        config (object): Configuration object containing hyperparameters.
            - num_blocks (int): Number of encoder blocks.
            - vocab_size (int): Size of the vocabulary.
            - final_dropout_prob (float): Dropout probability for the final layer.
            - d_model (int): Dimensionality of the model's hidden layers.
            - hidden_size (int): Size of the hidden embeddings.

    Attributes:
        num_blocks (int): Number of encoder blocks.
        vocab_size (int): Size of the vocabulary.
        final_dropout_prob (float): Dropout probability for the final layer.
        hidden_size (int): Size of the hidden embeddings.
        embed_layer (Embeddings): Embeddings layer.
        encoder (nn.ModuleList): List of encoder layers.
        dropout (nn.Dropout): Dropout layer for regularization.
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
        self.final_dropout_prob: float = config.final_dropout_prob
        self.hidden_size: int = config.hidden_size

        self.embed_layer: Embeddings = Embeddings(config)
        self.encoder: nn.ModuleList = nn.ModuleList([Encoder(config) for _ in range(self.num_blocks)])
        self.dropout: nn.Dropout = nn.Dropout(self.final_dropout_prob)
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

        if training:
            x_enc: torch.Tensor = self.dropout(x_enc)

        mlm_logits: torch.Tensor = self.mlm_prediction_layer(x_enc)
        nsp_logits: torch.Tensor = self.nsp_classifier(x_enc[:, 0, :])

        return self.softmax(mlm_logits), self.softmax(nsp_logits)
```

## End-to-End BERT

### LrSchedule

**Note:** The same one we used in the [Neural-Machine-Translator](https://github.com/AliHaiderAhmad001/Neural-Machine-Translator/tree/main/demo). But here we use PyTorch.

---

Before training the model, we need to determine the training strategy. In accordance with the paper "Attention Is All You Need" (same strategy used in BERT), we will utilize the Adam optimizer with a custom learning rate schedule. One technique we will employ is known as "learning rate warmup". This technique gradually increases the learning rate during the initial iterations of training in order to enhance stability and accelerate convergence. During the warmup phase, the learning rate is increased in a linear manner or according to a specific schedule until it reaches a predefined value. The objective of this warmup phase is to enable the model to explore a wider range of solutions during the initial training phase when the gradients might be large and unstable. The specific formula for the learning rate warmup is as follows:

```
learning_rate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
```

Here, `d_model` represents the dimensionality of the model, `step_num` indicates the current training step, and `warmup_steps` denotes the number of warmup steps. Typically, `warmup_steps` is set to a few thousand or a fraction of the total training steps. The motivation behind learning rate warmup is to address two challenges that often occur at the beginning of training:

1. **Large Gradient Magnitudes.** In the initial stages of training, the model parameters are randomly initialized, and the gradients can be large. If a high learning rate is applied immediately, it can cause unstable updates and prevent the model from converging. Warmup allows the model to stabilize by starting with a lower learning rate and gradually increasing it.
2. **Exploration of the Solution Space.** The model needs to explore a wide range of solutions to find an optimal or near-optimal solution. A high learning rate at the beginning may cause the model to converge prematurely to suboptimal solutions. Warmup enables the model to explore the solution space more effectively by starting with a low learning rate and then increasing it to search for better solutions.

```
import numpy as np
import torch.optim as optim

class ScheduledOptim:
    """
    A wrapper class for learning rate scheduling with an optimizer.

    Args:
        config (object): Configuration object containing hyperparameters.
            - n_warmup_steps (int): Number of warmup steps for learning rate.
            - d_model (int): Dimensionality of the model.

    Attributes:
        optimizer (torch.optim.Optimizer): The inner optimizer.
        n_warmup_steps (int): Number of warmup steps for learning rate.
        n_current_steps (int): Current number of steps.
        init_lr (float): Initial learning rate.
    """

    def __init__(self, config, optimizer):
        """
        Initialize the ScheduledOptim.
        """
        self.optimizer = optimizer
        self.n_warmup_steps: int = config.n_warmup_steps
        self.d_model = config.hidden_size
        self.n_current_steps: int = 0
        self.init_lr: float =  self.d_model** -0.5

    def step_and_update_lr(self):
        """Step with the inner optimizer and update learning rate."""
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer."""
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        """Calculate the learning rate scale based on the current steps."""
        return min([
            self.n_current_steps ** -0.5,
            (self.n_warmup_steps ** -1.5) * self.n_current_steps
        ])

    def _update_learning_rate(self):
        """Update learning rate based on the learning rate scale."""
        self.n_current_steps += 1
        lr: float = self.init_lr * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

```

### Training class

**BERTTrainer for Training and Testing BERT Models**

The provided code defines a `BERTTrainer` class, responsible for training and testing BERT (Bidirectional Encoder Representations from Transformers) models. BERT is a transformer-based model designed for natural language understanding and representation learning.

**Attributes:**

- **cuda_devices:** List of GPU devices for BERT training.
- **with_cuda:** Flag indicating whether to use CUDA for training.
- **log_freq:** Logging frequency during training.
- **batch_size:** Batch size for training and testing.
- **save_path:** Path for saving the model.
- **device:** Device (GPU or CPU) for training.
- **model:** BERT model.
- **train_data:** DataLoader for training data.
- **test_data:** DataLoader for testing data.
- **optim:** Optimizer for training.
- **criterion:** Negative Log Likelihood Loss function for predicting masked tokens.

**Methods:**

- **__init__(self, config: Config, bert: BERT, optim: ScheduledOptim, train_dataloader: DataLoader, test_dataloader: DataLoader = None):** Initializes the `BERTTrainer` class with configuration parameters, BERT model, optimizer, and data loaders.

- **train(self, epoch: int) -> None:** Trains the BERT model for one epoch.

- **test(self, epoch: int) -> None:** Tests the BERT model.

- **iteration(self, epoch: int, data_iter: DataLoader, train: bool = True) -> None:** Performs an iteration of training or testing.

- **process_batch(self, samples: List[dict]) -> dict:** Processes a batch of samples to create input tensors for the BERT model.

- **save(self, epoch: int) -> str:** Saves the current BERT model and optimizer state.

**Description:**

- The `BERTTrainer` class takes a BERT model, optimizer, and data loaders for training and testing.

- It supports both GPU and CPU training and can distribute training across multiple GPUs if available.

- The Negative Log Likelihood Loss function is used for predicting masked tokens during training.

- The `train` method trains the BERT model for one epoch, and the `test` method evaluates the model.

- The `iteration` method performs a single iteration of training or testing. It computes losses, updates parameters, and logs the progress.

- The `process_batch` method prepares a batch of samples for input to the BERT model by converting them into tensors.

- The `save` method saves the current state of the BERT model and optimizer to a specified path.

**Purpose:**

The primary purpose of the `BERTTrainer` class is to facilitate the training and testing of BERT models on natural language understanding tasks. It encapsulates the training logic, handling data, updating model parameters, and saving model checkpoints. This class is intended to be used as part of a larger system for developing and deploying BERT-based models for various NLP applications.

```
from typing import Tuple, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from scheduled_optim import ScheduledOptim
#from config import Config
#from bert_model import BERT

class BERTTrainer:
    """
    BERT Trainer class for training and testing BERT models.

    Args:
        config (Config): Configuration object containing parameters.
        bert (BERT): BERT model.
        optim (ScheduledOptim): Optimizer for training.
        train_dataloader (DataLoader): DataLoader for training data.
        test_dataloader (DataLoader, optional): DataLoader for testing data.

    Attributes:
        cuda_devices (List[int]): List of GPU devices for BERT training.
        with_cuda (bool): Whether to use CUDA for training.
        log_freq (int): Logging frequency during training.
        batch_size (int): Batch size for training and testing.
        save_path (str): Path for saving the model.
        device (torch.device): Device (GPU or CPU) for training.
        model (BERT): BERT model.
        train_data (DataLoader): DataLoader for training data.
        test_data (DataLoader): DataLoader for testing data.
        optim (ScheduledOptim): Optimizer for training.
        criterion (nn.NLLLoss): Negative Log Likelihood Loss function for predicting masked tokens.
    """

    def __init__(self, config: Config, bert: BERT, optim: ScheduledOptim,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None):
        self.cuda_devices: List[int] = config.cuda_devices
        self.with_cuda: bool = config.with_cuda
        self.log_freq: int = config.log_freq
        self.batch_size: int = config.batch_size
        self.save_path: str = config.save_path

        # Setup cuda device for BERT training
        cuda_condition: bool = torch.cuda.is_available() and self.with_cuda
        self.device: torch.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.model: BERT = bert.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if self.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for BERT" % torch.cuda.device_count())
            self.model: nn.DataParallel = nn.DataParallel(self.model, device_ids=self.cuda_devices)

        # Setting the train and test data loader
        self.train_data: DataLoader = train_dataloader
        self.test_data: DataLoader = test_dataloader
        self.optim: ScheduledOptim = optim

        # Using Negative Log Likelihood Loss function
        self.criterion: nn.NLLLoss = nn.NLLLoss(ignore_index=0)
        

        print("Total Parameters:", sum(p.nelement() for p in self.model.parameters()))

    def train(self, epoch: int) -> None:
        """
        Train the BERT model for one epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.iteration(epoch, self.train_data)

    def test(self, epoch: int) -> None:
        """
        Test the BERT model.

        Args:
            epoch (int): Current epoch number.
        """
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch: int, data_iter: DataLoader, train: bool = True) -> None:
        """
        Perform an iteration of training or testing.

        Args:
            epoch (int): Current epoch number.
            data_iter (DataLoader): DataLoader for the data.
            train (bool): Whether to train the model (True) or test (False).
        """
        str_code: str = "train" if train else "test"
        end_of_data: bool = False
        avg_loss: float = 0.0
        total_correct: int = 0
        total_element: int = 0
        i: int = 0

        buffer: List = []

        while not end_of_data:
            i += 1
            try:
                while len(buffer) < self.batch_size:
                    buffer.extend(next(data_iter))

            except StopIteration:
                end_of_data = True
                if len(buffer) == 0:
                    break

            current_batch = buffer[:self.batch_size]
            buffer = buffer[self.batch_size:]

            data = self.process_batch(current_batch)
            data = {key: value.to(self.device) for key, value in data.items()}

            mask_lm_output, next_sent_output = self.model.forward(data["bert_input"], data["segment_label"],
                                                                   training=train)

            next_loss = self.criterion(next_sent_output, data["is_next"])
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
            loss = next_loss + mask_loss

            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step_and_update_lr()

            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                output_str = "Epoch: {}, Iteration: {}, Avg Loss: {:.4f}, Avg Acc: {:.2f}%, Current Loss: {:.4f}".format(
                    post_fix['epoch'], post_fix['iter'], post_fix['avg_loss'], post_fix['avg_acc'], post_fix['loss']
                )
                print(output_str)

        print("Epoch %d, %s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def process_batch(self, samples: List[dict]) -> dict:
        """
        Process a batch of samples to create input tensors for the BERT model.

        Args:
            samples (List[dict]): List of dictionaries, where each dictionary contains:
                - 'bert_input': List[int] - List of tokenized input for BERT.
                - 'bert_label': List[int] - List of tokenized labels for BERT.
                - 'segment_label': List[int] - List of segment labels.
                - 'is_next': int - Binary indicator for next sentence prediction.

        Returns:
            dict: A dictionary containing processed input tensors:
                - 'bert_input': torch.Tensor - Tensor of input tokens.
                - 'bert_label': torch.Tensor - Tensor of label tokens.
                - 'segment_label': torch.Tensor - Tensor of segment labels.
                - 'is_next': torch.Tensor - Tensor of binary indicators.
        """
        bert_input_list = []
        bert_label_list = []
        segment_label_list = []
        is_next_list = []

        for sample in samples:
            bert_input_list.append(sample['bert_input'])
            bert_label_list.append(sample['bert_label'])
            segment_label_list.append(sample['segment_label'])
            is_next_list.append(sample['is_next'])

        # Convert lists to tensors
        bert_input_tensor = torch.tensor(bert_input_list)
        bert_label_tensor = torch.tensor(bert_label_list)
        segment_label_tensor = torch.tensor(segment_label_list)
        is_next_tensor = torch.tensor(is_next_list)

        output = {
            "bert_input": bert_input_tensor,
            "bert_label": bert_label_tensor,
            "segment_label": segment_label_tensor,
            "is_next": is_next_tensor
        }

        return output

    def save(self, epoch: int) -> str:
        """
        Save the current BERT model and optimizer state.

        Args:
            epoch (int): Current epoch number.

        Returns:
            str: Path to the saved model.
        """
        output_path: str = self.save_path + ".ep%d" % epoch

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, output_path)

        self.model.to(self.device)

        print("Epoch %d Model and Optimizer State Saved on:" % epoch, output_path)
        return output_path
```

### Monitor

The provided script is designed for training BERT models on custom text datasets. It involves loading a dataset, initializing a BERT model, creating data loaders, and performing training and testing loops.

**Key Components:**

1. **Setting Seeds (`set_seeds` function):**
   - Sets random seeds for reproducibility across runs.
   - Utilizes `random`, `numpy`, and `torch` seeds to ensure consistency in random processes.

2. **Run Function (`run` function):**
   - The main function orchestrating the BERT training process.

3. **Loading Train Dataset:**
   - Utilizes the `CustomTextDataset` class to load the training dataset. This dataset is specifically designed for BERT-style pre-training on text data.

4. **Loading Test Dataset (Optional):**
   - If a test dataset is provided in the configuration (`config.test_dataset`), it is loaded using the same `CustomTextDataset` class.

5. **BERT Model Initialization:**
   - Initializes a BERT model (`bert`) using the `BERT` class, which is presumably defined elsewhere in the codebase. The model architecture and configuration are determined by the `Config` parameters.

6. **Data Loaders Creation:**
   - Creates data loaders for the training and test datasets using PyTorch's `DataLoader`. The `worker_init_fn` is set to a seed for additional reproducibility.

7. **Optimizer and Scheduler Initialization:**
   - Initializes an Adam optimizer for training the BERT model. The learning rate and other parameters are specified in the `Config`.
   - Uses a scheduler (`ScheduledOptim`) to adjust the learning rate during training.

8. **BERT Trainer Initialization:**
   - Initializes a `BERTTrainer` object, which is responsible for handling the training and testing loops. The trainer takes the BERT model, optimizer, and data loaders.

9. **Training Loop:**
   - Iterates over a specified number of epochs (`config.epochs`).
   - Calls the `train` method of the `BERTTrainer` to train the model for each epoch.
   - Saves the model checkpoint after each epoch using the `save` method of the `BERTTrainer`.

10. **Testing Loop (Optional):**
    - If a test dataset is provided, calls the `test` method of the `BERTTrainer` after each epoch to evaluate the model on the test set.

11. **Configuration Setup and Execution (`if __name__ == "__main__":`):**
    - Defines a configuration (`config`) using the `Config` class, specifying various parameters such as sequence length, learning rate, batch size, etc.

12. **Run BERT Training (`run(config)`):**
    - Executes the BERT training process by calling the `run` function with the provided configuration.

**Purpose:**

The script serves the purpose of training BERT models on custom text datasets. It encapsulates the entire training pipeline, from loading datasets to initializing models, creating data loaders, and executing training and testing loops. The modular design allows users to easily configure and adapt the script for different datasets and experimental settings. The inclusion of seed-setting mechanisms contributes to result reproducibility.

```
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
#from scheduled_optim import ScheduledOptim
import random as rd
import numpy as np
import argparse
#from dataset import CustomTextDataset
#from bert_trainer import BERTTrainer


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
    print(train_dataset.get_filenames)
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
    config = Config(
        prop=0.15,
        tokenizer_path='bert-base-uncased',
        seq_len= 128,
        delimiters= '?.!;:',
        lower_case= True,
        buffer_size= 10,
        shuffle=True,
        data_dir= 'datasets/aclImdb',
        hidden_size= 64,
        vocab_size= 30522,
        hidden_dropout_prob= 0.1,
        num_heads= 8,
        num_blocks= 2,
        final_dropout_prob= 0.5,
        n_warmup_steps= 4000,
        weight_decay= 0.01,
        lr= 1e-4,
        betas= (0.9, 0.999),
        cuda_devices=None,
        with_cuda= True,
        log_freq= 10,
        batch_size= 32,
        save_path= 'tmp',
        seed= 2023,
        test_dataset= None,
        epochs= 2
    )


    # Run BERT training
    run(config)
```

