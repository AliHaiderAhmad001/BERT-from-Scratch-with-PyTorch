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
