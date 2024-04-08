# BERT-from-Scratch-with-PyTorch 

BERT, which stands for "Bidirectional Encoder Representations from Transformers," is a groundbreaking natural language processing (NLP) model that has had a profound impact on a wide range of NLP tasks. It was introduced by researchers at Google AI in their paper titled "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)" in 2018.


---

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Ubuntu-orange.svg)](https://www.ubuntu.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
![Transformers](https://img.shields.io/badge/transformers-4.36-yellow.svg)
[![Python](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/)


# Limitations

The goal of this project is to provide a deep understanding of the BERT architecture and its inner workings. So, it's mainly for educational purposes. You can fully understand the structure and working mechanism of this model here, and use the components I have implemented in your projects. Generally, if you want to use the project to train your language model with big data, you may need to modify the dataset file to be able to process big data more efficiently. I designed the dataset file mainly to handle simple, not large, data, because I am not in this regard now.

This codebase is set up for model pre-training. To use this implementation for fine-tuning on specific tasks, further modifications may be required.


## Project Overview

Welcome to "BERT-from-Scratch-with-PyTorch"! This project is an ambitious endeavor to create a BERT model from scratch using PyTorch. My goal is to provide an in-depth and comprehensive resource that helps enthusiasts, researchers, and learners gain a precise understanding of BERT, from its fundamental concepts to the implementation details.


**Project Objectives**

1. **Educational Resource**: My primary objective is to create a detailed educational resource that demystifies BERT and the Transformer architecture. I aim to provide step-by-step explanations and code demonstrations for each component of the model, making it accessible to anyone interested in natural language processing and deep learning.

2. **Hands-On Learning**: I encourage hands-on learning. By building the BERT model from the ground up, users can grasp the core principles behind BERT's pretraining and fine-tuning phases. I believe that understanding the model's inner workings is the key to harnessing its power effectively.

3. **Open Source and Accessible**: I'm committed to making this project open source and freely accessible to the community. Learning should be accessible to all, and I aim to contribute to the open-source AI and NLP communities.

4. **On a personal level**: Although I am very familiar with this model and how it works, I had not built it from scratch nor trained models using the concept of masked language modeling (MLM) before. So I'll get my hands dirty a little, it will increase my understanding of it and maybe give me new ideas.

**Key Features**

- **Step-by-Step Implementation**: We break down the BERT model into its constituent parts, guiding you through the implementation of each component.

- **Detailed Explanations**: We provide comprehensive explanations of the underlying concepts, ensuring you grasp not just the "how" but also the "why."

- **Demo and Examples**: Code demonstrations are accompanied by practical examples, making it easier to apply your newfound knowledge to real-world problems.

- **Extensible**: The codebase is designed to be extensible. You can use it as a foundation to experiment with variations of BERT or adapt it for specific NLP tasks.

So, let the games begin...

**Getting Started**

To get started with building BERT from scratch, you must have a comprehensive understanding of transformers (this is very essential in my view). The good news is that I previously built this model from scratch with full explanations you can find [here](https://github.com/AliHaiderAhmad001/Neural-Machine-Translator/blob/main/README.md).

## Documentation
You can read the full explanation of all components in [Demo](https://github.com/AliHaiderAhmad001/BERT-from-Scratch-with-PyTorch/tree/main/demo).

## Usage
1. Clone the repository:

   ```bash
   git clone https://github.com/AliHaiderAhmad001/BERT-from-Scratch-with-PyTorch.git
   cd BERT-from-Scratch-with-PyTorch
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install project dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Download and prepare Dataset: You can go and review the demo.You can work on the same dataset, change it or adjust your preferences. However, You can download the dataset directly from [here](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).

```bash
cd 'datasets'
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
cd ..
```
   
6. Optionally, you can re-train  WordPiece tokenizer:
   ```bash
   python word_piece_trainer.py --tokenizer_name bert-base-cased --data_dir aclImdb --batch_size 1000 --vocab_size 30522 --save_fp tokenizer/adapted-tokenizer
   ```
7. Train the BERT model:
   ```bash
    python run.py --prop 0.15 --tokenizer_path bert-base-uncased --seq_len 512 --delimiters ".,;:!? " --lower_case True --buffer_size 1 --shuffle True --data_dir 'datasets/aclImdb' --hidden_size 768 --vocab_size 30522 --hidden_dropout_prob 0.1 --num_heads 8 --num_blocks 12 --final_dropout_prob 0.5 --n_warmup_steps 10000 --weight_decay 0.01 --lr 1e-4 --betas 0.9 0.999 --with_cuda True --log_freq 10 --batch_size 64 --save_path 'tmp' --seed 2023 --epochs 10

   ```
   

## Contributing

If you find some bug or typo, please let me know or fix it and push it to be analyzed.

```markdown
# Contribution Guidelines
- Fork the project.
- Create a new branch for your feature or bug fix.
- Make your changes and test thoroughly.
- Submit a pull request with a clear description of your changes.
```

## Acknowledgements

This project was inspired by the works of:
* [jacobdevlin](https://github.com/jacobdevlin-google) on [bert](https://github.com/google-research/bert).
* [Junseong Kim](https://github.com/codertimo/BERT-pytorch/commits?author=codertimo) on [BERT-pytorch](https://github.com/codertimo/BERT-pytorch).

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/AliHaiderAhmad001/Neural-Machine-Translator/blob/main/LICENSE.txt) file for details.

---
