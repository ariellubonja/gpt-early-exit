from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer
from torch import nn


def plot_intermediate_model_outputs(model, xlim, bins=100, ylim=None, num_layers=None, keys = ['initial_hidden_states', 'post_ln1_hidden_states', 'attn_projection_output', 'post_attn_residual_hidden_states', 'post_cross_attn_hidden_states', 'post_ln2_hidden_states', 'post_feed_fwd_hidden_states', 'post_feed_fwd_residual_hidden_states']):
    """
    Plots the intermediate outputs of the model for each layer.

    Args:
    - model: The model whose intermediate outputs are to be plotted.
    - bins: Number of bins for the histogram. FUNCTION USES FIXED BINS IN THE XLIM RANGE!
    - xlim: Tuple of two values specifying the range of the x-axis for the histogram.
    - ylim: Tuple of two values specifying the range of the y-axis for the histogram.
    - num_layers: Number of Transformer layers layers to plot. If None, all layers are plotted.
    - keys: List of keys for which intermediate outputs are to be plotted. Default is 
    ['initial_hidden_states', 'post_ln1_hidden_states', 'attn_projection_output', 'post_attn_residual_hidden_states', 'post_cross_attn_hidden_states', 'post_ln2_hidden_states', 'post_feed_fwd_hidden_states', 'post_feed_fwd_residual_hidden_states']
    """

    if num_layers is None:
        num_layers = len(model.h)

    # Create fixed bins within the xlim range
    # If you set nr. bins, it either maintains the slice across images, or it concentrates or widens
    # Bins depending on data range. This makes plots inconsistent.
    # Fixed bins should fix that, at the cost of missing whatever data lies outside the range
    bin_edges = np.linspace(xlim[0], xlim[1], bins+1)
    
    for i in range(num_layers):
        int_out = model.h[i].intermediate_outputs
        for key in keys:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            current_hidden_layer = int_out[key].squeeze()

            im = ax1.imshow(current_hidden_layer)#, aspect='auto', interpolation='nearest')
            ax1.set_title(f'Layer {i} {key} - Image')
            ax1.figure.colorbar(im, ax=ax1)
            counts, bins, _ = ax2.hist(current_hidden_layer.ravel(), bins=bin_edges)#bins='auto')#, bins=20, color='blue')

            # assert sum(counts) == int_out[key][0].numel(), "Histogram counts do not match the number of elements in the tensor."
            # if sum(counts) != int_out[key][0].numel():
            #     ariel_debug = 1

            ax2.set_xlim(xlim)  # Make histogram range consistent. Necessary to determine bins too
            if ylim is not None:
                ax2.set_ylim(ylim)
            ax2.set_title(f'Layer {i} {key} - Histogram')
            plt.show()


def compute_model_perplexity(model, dataset: list[str], n_rows: int = None, batch_size=4, tokenizer=GPT2Tokenizer.from_pretrained("gpt2-medium")):
    # We need a padding token for DataLoader
    tokenizer.pad_token = tokenizer.eos_token
    model = model.eval()

    # If n_rows is specified, slice the dataset
    if n_rows is not None:
        text_data = text_data[:n_rows]
    else:
        text_data = dataset

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


def filter_state_dict(original_state_dict, max_layer):
    """
    Filter the state dictionary to only include layers up to 'max_layer'.
    """
    filtered_state_dict = {}
    for key, value in original_state_dict.items():
        # Assuming layer numbers are specified in the keys like 'h.0.', 'h.1.', etc.
        if "h." in key:
            layer_number = int(key.split('.')[1])
            if layer_number <= max_layer:
                filtered_state_dict[key] = value
        else:
            # Include all other parameters that do not depend on layer number
            filtered_state_dict[key] = value
    return filtered_state_dict



def early_exit_generation(model, input_text, tokenizer, num_tokens_to_generate=20, exit_layer=5):
    def get_hiddenstates_attn(input_ids, model, tokenizer):
        with torch.no_grad():
            output = model(input_ids=input_ids)
        hidden_states = output.hidden_states
        attentions = output.attentions
        return hidden_states, attentions

    token_probabilities = []
    logits = []
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    for _ in range(num_tokens_to_generate):
        # outputs = model(input_ids)
        hidden_states, attns = get_hiddenstates_attn(input_ids, model, tokenizer)

        ln_f = nn.LayerNorm(model.config.n_embd, eps=model.config.layer_norm_epsilon)
        layer_norm = ln_f(hidden_states[exit_layer])

        # if apply_layer_norm:
        if exit_layer != len(hidden_states) - 1:
            # print("yes for ", str(exit_layer))
            ln_f = nn.LayerNorm(model.config.n_embd, eps=model.config.layer_norm_epsilon)
            layer_norm = ln_f(hidden_states[exit_layer])
            logits_out = model.lm_head(layer_norm)
        else:
            # print("no layernorm or lm_head for ", str(exit_layer))
            # logits = model.lm_head(hidden_states[exit_layer])
            logits_out = model.lm_head(hidden_states[exit_layer])

        # Only use the logits from the last token position
        next_token_logits = logits_out[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        k = 15
        top_k_tokens = torch.topk(next_token_logits, k, dim=-1)
        top_k_probabilities = F.softmax(top_k_tokens.values, dim=-1)
        
        plt.figure()
        plt.plot(top_k_probabilities[0].tolist())
        plt.title('Top-15 probabilities for early exit layer ' + str(exit_layer))
        
        # plt.legend()

        # print(next_token_logits)

        # Append the predicted token ID to the input sequence
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # get the probabilities
        probabilities = F.softmax(next_token_logits, dim=-1)
        top_proba = probabilities[0][next_token_id].item()
        # top_proba = next_token_logits[0][next_token_id].item()
        token_probabilities.append(top_proba)
        logits.append(next_token_logits)

    # print(input_ids)
    generated_text = tokenizer.decode(input_ids[0])

    return generated_text, token_probabilities