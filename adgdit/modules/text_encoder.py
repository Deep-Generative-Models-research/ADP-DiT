import torch
import torch.nn as nn
from transformers import BertForMaskedLM

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        
        # BertForMaskedLM을 불러옵니다.
        self.text_encoder = BertForMaskedLM.from_pretrained('path/to/bert_masked_lm')
        
        # 기존 가중치 가져오기
        self.embedding_weights = self.text_encoder.bert.embeddings.word_embeddings.weight.data
        
        # 새로운 임베딩 초기화 (vocab_size 변경이 필요한 경우)
        new_vocab_size = 28996
        old_vocab_size, embedding_dim = self.embedding_weights.shape
        
        # 새로운 임베딩 초기화
        if new_vocab_size < old_vocab_size:
            print(f"Resizing embedding weights from {old_vocab_size} to {new_vocab_size}")
            self.embedding_weights = self.embedding_weights[:new_vocab_size, :]
        else:
            print(f"Initializing new embedding weights for extra {new_vocab_size - old_vocab_size} tokens")
            extra_weights = torch.randn(new_vocab_size - old_vocab_size, embedding_dim) * 0.02  # 초기화
            self.embedding_weights = torch.cat([self.embedding_weights, extra_weights], dim=0)
        
        # 새로운 임베딩을 설정합니다.
        self.text_encoder.bert.embeddings.word_embeddings = nn.Embedding.from_pretrained(self.embedding_weights, freeze=False)
        
    def forward(self, input_ids, attention_mask=None):
        # BertForMaskedLM의 forward 함수 사용
        outputs = self.text_encoder.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
