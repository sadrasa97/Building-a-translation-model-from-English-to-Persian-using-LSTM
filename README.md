
# Neural Machine Translation (English-Persian)

This project implements a Neural Machine Translation (NMT) system for translating English to Persian. It uses a variety of NLP techniques, including vocabulary generation, tokenization, and deep learning models, for sequence-to-sequence translation.

---

## Features

- Preprocessing pipeline for English-Persian datasets.
- Vocabulary generation using MBart tokenizer (English) and SentencePiece (Persian).
- Embedding initialization using Word2Vec for both languages.
- PyTorch-based implementation of sequence-to-sequence translation models.
- Custom `DataLoader` with batching and padding for efficiency.
- Support for GPU and CPU training.

---

## Installation

To run this project, ensure you have Python 3.8 or later and install the required libraries using `pip`:

```bash
pip install torch transformers datasets sentencepiece gensim scikit-learn tqdm
```

---

## Dataset

This project uses the [Global Voices English-Persian translation dataset](https://huggingface.co/datasets/persiannlp/parsinlu_translation_en_fa). The dataset is processed into training, validation, and test splits.

---

## Usage

1. **Download Dataset**  
   The dataset is loaded and sampled directly using the `datasets` library.

2. **Preprocess the Data**  
   - Train, validation, and test splits are created.
   - Custom vocabulary is built using MBart for English and SentencePiece for Persian.

3. **Train the Model**  
   - Data is fed into a sequence-to-sequence model using custom `DataLoader`.
   - Pre-trained Word2Vec embeddings are used for both English and Persian vocabularies.

4. **Run Translation**  
   - Translate sentences from English to Persian using the trained model.

---

## Code Overview

### 1. Preprocessing
- **Tokenization**:  
  Tokenizers are implemented for both English and Persian using MBart and SentencePiece, respectively.
  
- **Vocabulary Building**:  
  Vocabulary is built from training data, including special symbols like `<unk>`, `<pad>`, `<bos>`, and `<eos>`.

- **Data Loading**:  
  Data is loaded using custom PyTorch datasets and batched using a custom collate function.

### 2. Embeddings
- Word2Vec embeddings are initialized for both English and Persian using pre-trained Word2Vec models.
- The embedding matrix is mapped to the vocabulary.

### 3. Model Training
- The data is prepared and loaded into DataLoaders for training and evaluation.
- Batch processing with padding ensures consistent input sizes.

---

## Example

To train the model, modify the `batch_size` and other hyperparameters as needed:

```python
train_dataloader, valid_dataloader, test_dataloader = get_translation_dataloaders(batch_size=8)
```

After training, test the model by translating sample sentences:

```python
english_sentence = "I love books."
# Model prediction here
persian_translation = "من کتاب‌ها را دوست دارم."
```

---

## Requirements

- Python 3.8+
- Libraries:
  - PyTorch
  - Transformers
  - Datasets
  - SentencePiece
  - Gensim
  - scikit-learn
  - tqdm

---

## Future Work

- Add support for additional language pairs.
- Implement more advanced NMT models like Transformer-based architectures.
- Incorporate BLEU score evaluation for translation quality.
- Explore fine-tuning pre-trained multilingual models like MBart.

---

## License

This project is open-sourced under the MIT License.

---

## Acknowledgments

- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [MBart](https://huggingface.co/transformers/model_doc/mbart.html)
- [SentencePiece](https://github.com/google/sentencepiece)
- [Word2Vec](https://radimrehurek.com/gensim/)

```
