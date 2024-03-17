import torch
from dataclasses import dataclass
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer


@dataclass
class ModelConfig:
    tokenizer_path: str = "bert-base-uncased"


@dataclass
class DataConfig:
    eval_file_path: str = './data/validation_dataset.csv'
    model_save_path: str = './checkpoint'


def main() -> None:
    sentences = pd.read_csv(DataConfig.eval_file_path)['question'].values()
    tokenizer = BertTokenizer.from_pretrained(ModelConfig.tokenizer_path, do_lower_case=True)

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
    segments_ids = [0] * len(tokenized_texts)
    segments_tensors = torch.tensor([segments_ids])
    test_dataloader = DataLoader(torch.tensor([indexed_tokens]))

    model = torch.load(DataConfig.model_save_path)

    model.eval()
    predictions = []
    answer = []
    for tokenized_text in test_dataloader:
        logit = model(tokenized_text, token_type_ids=None,
                      attention_mask=segments_tensors)
        logit = logit.detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logit, axis=2)])
        for k, j in enumerate(predictions[0]):
            answer.append(tokenizer.convert_ids_to_tokens(tokenized_text[0].to('cpu').numpy())[k])


if __name__ == "__main__":
    main()
