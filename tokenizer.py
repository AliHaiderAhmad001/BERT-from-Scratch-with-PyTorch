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

