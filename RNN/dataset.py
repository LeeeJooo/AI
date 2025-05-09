import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class Dataset:
    def __init__(self):
        self.dataset = self._load_imdb()
    
    def _load_imdb(self):
        _dataset = load_dataset("imdb")
        _shuffled_dataset = _dataset.shuffle(seed=42)
        return _shuffled_dataset
    
    @staticmethod
    def preprocess_imdb(config_obj, dataset):
        tokenizer = AutoTokenizer.from_pretrained(config_obj.database_config.imdb_tokenizer)

        train_dataset = dataset['train']
        train_dataset_size = int(len(train_dataset) * config_obj.database_config.imdb_train_valid_ratio)
        train_dataset_text = train_dataset['text']
        train_dataset_text_tokenized = tokenizer(train_dataset_text,    # 입력 문장
                                                 return_tensors="pt",   # PyTorch 텐서로 반환
                                                 padding=True,          # 시퀀스 패딩
                                                 truncation=True        # max_length 넘으면 자름
                                                 )
        train_x = train_dataset_text_tokenized['input_ids'][:train_dataset_size].to(config_obj.device)
        train_y = torch.tensor(train_dataset['label'][:train_dataset_size], dtype=torch.int64).to(config_obj.device)
        train_x = train_x.reshape(config_obj.pipeline_config.batch_size)
        valid_x = train_dataset_text_tokenized['input_ids'][train_dataset_size:].to(config_obj.device)
        valid_y = torch.tensor(train_dataset['label'][train_dataset_size:], dtype=torch.int64).to(config_obj.device)

        test_dataset = dataset['test']
        test_dataset_text = test_dataset['text']
        test_dataset_text_tokenized = tokenizer(test_dataset_text,      # 입력 문장
                                                 return_tensors="pt",   # PyTorch 텐서로 반환
                                                 padding=True,          # 시퀀스 패딩
                                                 truncation=True        # max_length 넘으면 자름
                                                 )
        test_x = test_dataset_text_tokenized['input_ids'].to(config_obj.device)
        test_y = torch.tensor(test_dataset['label'], dtype=torch.int64).to(config_obj.device)
        
        return tokenizer.vocab_size, train_x, train_y, valid_x, valid_y, test_x, test_y

if __name__=="__main__":
    from config import Config
    config_obj = Config()
    dataset_obj = Dataset()
    vocab_size, train_x, train_y, valid_x, valid_y, test_x, test_y = dataset_obj.preprocess_imdb(config_obj, dataset_obj.dataset)