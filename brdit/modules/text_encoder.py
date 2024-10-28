import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class CLIPTextEmbedder(nn.Module):
    available_models = ["openai/clip-vit-base-patch32"]  # Specify available model

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=77  # Default CLIP text length limit
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        # Load the model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_tokens_and_mask(self, texts):
        # Tokenize the input texts
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        return inputs.input_ids, inputs.attention_mask

    def get_text_embeddings(self, texts):
        tokens, attention_mask = self.get_tokens_and_mask(texts)
        with torch.no_grad():
            # Generate embeddings from the text encoder
            text_embeddings = self.model.get_text_features(
                input_ids=tokens,
                attention_mask=attention_mask
            )
        return text_embeddings

    @torch.no_grad()
    def __call__(self, texts):
        # Directly process texts through get_text_embeddings
        return self.get_text_embeddings(texts)

# Example usage:

# Initialize the CLIP text embedder
embedder = CLIPTextEmbedder()

# Example input text prompts
texts = ["A cute husky", "A cat sitting on a chair"]

# Generate text embeddings using CLIP
text_embeddings = embedder(texts)

print("Text embeddings generated:", text_embeddings)
