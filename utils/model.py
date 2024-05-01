from matplotlib import pyplot as plt
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer

def plot_intermediate_model_outputs(model, keys = ['initial_hidden_states', 'post_ln1_hidden_states', 'attn_projection_output', 'post_attn_residual_hidden_states', 'post_cross_attn_hidden_states', 'post_ln2_hidden_states', 'post_feed_fwd_hidden_states', 'post_feed_fwd_residual_hidden_states'], bins=100):
    for i in range(len(model.h)):
        int_out = model.h[i].intermediate_outputs
        for key in keys:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            im = ax1.imshow(int_out[key][0])#, aspect='auto', interpolation='nearest')
            ax1.set_title(f'Layer {i} {key} - Image')
            ax1.figure.colorbar(im, ax=ax1)
            ax2.hist(int_out[key][0], bins)#.ravel())#, bins=20, color='blue')
            ax2.set_title(f'Layer {i} {key} - Histogram')
            plt.show()


def compare_model_perplexity(model1, model2, dataset_name: str = "Trelis/tiny-shakespeare", n_rows: int = None, batch_size=4):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    # We need a padding token for DataLoader
    tokenizer.pad_token = tokenizer.eos_token
    model1 = model1.eval()
    model2 = model2.eval()

    dataset = load_dataset(dataset_name)
    text_data = dataset['train']['Text']

    # If n_rows is specified, slice the dataset
    if n_rows is not None:
        text_data = text_data[:n_rows]

    # Tokenization
    def encode(text):
        return tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")['input_ids']
    
    input_ids = torch.cat([encode(text) for text in text_data], dim=0)

    # Create DataLoader for batch processing
    loader = DataLoader(input_ids, batch_size=batch_size, shuffle=False)

    # Initialize variables to compute perplexity
    total_loss_original = 0
    total_loss_modified = 0
    total_items = 0

    # Disable gradients for evaluation
    with torch.no_grad():
        for batch in loader:
            model1_outputs = model1(batch)
            model2_outputs = model2(batch)

            # Calculate loss for the batch
            shift_logits_original = model1_outputs.logits[..., :-1, :].contiguous()
            shift_labels_original = batch[..., 1:].contiguous()
            loss_original = F.cross_entropy(shift_logits_original.view(-1, shift_logits_original.size(-1)), shift_labels_original.view(-1))

            shift_logits_modified = model2_outputs.logits[..., :-1, :].contiguous()
            shift_labels_modified = batch[..., 1:].contiguous()
            loss_modified = F.cross_entropy(shift_logits_modified.view(-1, shift_logits_modified.size(-1)), shift_labels_modified.view(-1))

            total_loss_original += loss_original.item() * batch.size(0)
            total_loss_modified += loss_modified.item() * batch.size(0)
            total_items += batch.size(0)

    # Calculate perplexity
    perplexity_original = np.exp(total_loss_original / total_items)
    perplexity_modified = np.exp(total_loss_modified / total_items)

    print(f"Perplexity of Original Model: {perplexity_original}")
    print(f"Perplexity of Modified Model: {perplexity_modified}")


def compute_model_perplexity(model, dataset_name: str = "Trelis/tiny-shakespeare", n_rows: int = None, batch_size=4):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    # We need a padding token for DataLoader
    tokenizer.pad_token = tokenizer.eos_token
    model = model.eval()

    dataset = load_dataset(dataset_name)
    text_data = dataset['train']['Text']  # TODO Add test data too. we're not doing any training anyway

    # If n_rows is specified, slice the dataset
    if n_rows is not None:
        text_data = text_data[:n_rows]

    # Tokenization
    def encode(text):
        return tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")['input_ids']
    
    input_ids = torch.cat([encode(text) for text in text_data], dim=0)

    # Create DataLoader for batch processing
    loader = DataLoader(input_ids, batch_size=batch_size, shuffle=False)

    # Initialize variables to compute perplexity
    total_loss_original = 0
    total_items = 0

    # Disable gradients for evaluation
    with torch.no_grad():
        for batch in loader:
            model1_outputs = model(batch)

            # Calculate loss for the batch
            shift_logits_original = model1_outputs.logits[..., :-1, :].contiguous()
            shift_labels_original = batch[..., 1:].contiguous()
            loss_original = F.cross_entropy(shift_logits_original.view(-1, shift_logits_original.size(-1)), shift_labels_original.view(-1))

            total_loss_original += loss_original.item() * batch.size(0)
            total_items += batch.size(0)

    # Calculate perplexity
    perplexity_original = np.exp(total_loss_original / total_items)

    print(f"Perplexity of Model: {perplexity_original}")
    return perplexity_original