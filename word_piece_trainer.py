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

    def train(self, data_dir: str, batch_size: int, vocab_size: int, save: bool, save_fp: str) -> None:
        """
        Train the tokenizer on a new corpus.

        Args:
            data_dir (str): Corpus directory path.
            batch_size (int): Batch size for reading files.
            vocab_size (int): Target vocabulary size.
            save (bool): Whether to save the adapted tokenizer.
            save_fp (str): File path to save the tokenizer.

        """
        training_corpus = self.read_batch_of_files(data_dir, batch_size)
        self.tokenizer = self.tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        if save:
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
        Get a list of filenames in a directory.

        Args:
            data_dir (str): Directory path.

        Returns:
            list: List of filenames.

        """
        filenames = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                filenames.append(os.path.join(root, file))
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
    parser = argparse.ArgumentParser(description="Tokenizer Training Script")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-cased", help="Name of the pretrained tokenizer.")
    parser.add_argument("--data_dir", type=str, default="aclImdb", help="Directory containing the training data.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for reading files.")
    parser.add_argument("--vocab_size", type=int, default=30622, help="Target vocabulary size.")
    parser.add_argument("--save", action="store_true", help="Whether to save the adapted tokenizer.")
    parser.add_argument("--save_fp", type=str, default="tokenizer/adapted-tokenizer", help="File path to save the tokenizer.")
    
    args = parser.parse_args()

    wp_trainer = WordPieceTrainer(tokenizer_name=args.tokenizer_name)
    wp_trainer.train(data_dir=args.data_dir, batch_size=args.batch_size, vocab_size=args.vocab_size, save=args.save, save_fp=args.save_fp)

if __name__ == "__main__":
    main()

