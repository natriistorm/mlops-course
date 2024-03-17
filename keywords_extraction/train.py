from dataclasses import dataclass
import pandas as pd
from tqdm import trange
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForTokenClassification


@dataclass
class ModelConfig:
    model_path: str = "bert-base-uncased"
    tokenizer_path: str = "bert-base-uncased"


@dataclass
class TokenizerConfig:
    max_seq_length: int = 500
    dtype: str = "long"
    truncating: str = "post"
    padding: str = "post"


@dataclass
class TrainingConfig:
    weight_decay_rate: float = 0.01
    learning_rate: float = 0.01
    epochs: int = 10
    max_grad_norm: float = 1.0
    batch_size: int = 8


@dataclass
class DataConfig:
    train_file_path: str = './data/stack_overflow_questions.csv'
    model_save_path: str = './checkpoint'


def main() -> None:
    train_path = DataConfig.train_file_path

    sentences = pd.read_csv(train_path)['question'].values()
    labels = pd.read_csv(train_path)['tags'].values()
    tokenizer = BertTokenizer.from_pretrained(ModelConfig.tokenizer_path, do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    tokenized_tags = [tokenizer.tokenize(label) for label in labels]
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=TokenizerConfig.max_seq_length,
                              dtype=TokenizerConfig.dtype,
                              truncating=TokenizerConfig.truncating,
                              padding=TokenizerConfig.padding)

    tags = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_tags],
                         maxlen=TokenizerConfig.max_seq_length,
                         dtype=TokenizerConfig.dtype,
                         truncating=TokenizerConfig.truncating,
                         padding=TokenizerConfig.padding)

    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=2018, test_size=0.1)

    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)

    train_data = TensorDataset(torch.tensor(tr_inputs), torch.tensor(tr_masks), torch.tensor(tr_tags))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TrainingConfig.batch_size)

    model = BertForTokenClassification.from_pretrained(ModelConfig.model_path)
    model_path = DataConfig.model_save_path

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': TrainingConfig.weight_decay_rate
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0
        }
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    print("Training model...")
    for _ in trange(TrainingConfig.epochs, desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=TrainingConfig.max_grad_norm)
            optimizer.step()
            model.zero_grad()
    torch.save(model.state_dict(), model_path + ".pth")
    print(f"Model saved to {DataConfig.model_save_path}")


if __name__ == "__main__":
    main()
