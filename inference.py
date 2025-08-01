import os
import torch
import torch.nn.functional as F


def causal_mask(size):
    """Look ahead mask or attention mask"""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def top_k_sampling(logits, k):
    top_k_probs, top_k_indices = torch.topk(logits, k, dim=-1)
    top_k_probs = F.softmax(top_k_probs, dim=-1)
    sampled_index = torch.multinomial(top_k_probs, 1)
    next_word = top_k_indices.gather(-1, sampled_index)
    return next_word.squeeze()

def decode(model, max_len, device, k=5):
    """returns the decoded tensor (token ids)"""
    cls_idx, sep_idx = 15050,15048
    model_input = torch.empty(1, 1, dtype=torch.int64).fill_(cls_idx).to(device)

    while model_input.size(1) < max_len:
        decoder_mask = causal_mask(model_input.size(1)).to(torch.int64).to(device)
        out = model(model_input, decoder_mask)
        logits = out[:, -1, :]  # Get the logits of the last token
        # Use top-k sampling to select the next word
        next_word = top_k_sampling(logits, k)
        if next_word == sep_idx:
            break
        model_input = torch.cat([model_input, next_word.view(1, 1).to(device)], dim=1)
    return model_input.squeeze(0)[1:]

def run_validation(model, k, custom_tokenizer, max_len, device, print_msg):
    model.to(device)
    model.eval()

    try:
      console_width = os.get_terminal_size().columns
    except OSError:
      console_width = 80

    model_out = decode(model, max_len, device, k)
    model_out_text = custom_tokenizer.decode(model_out.detach().cpu().numpy(), skip_special_tokens=True)
    print_msg(f"{'GENERATED:':>12}{model_out_text}\n{'-'*console_width}")