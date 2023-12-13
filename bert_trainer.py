from typing import Tuple, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scheduled_optim import ScheduledOptim
from config import Config
from bert_model import BERT

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
