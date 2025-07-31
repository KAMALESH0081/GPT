import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    def __init__(self, dataframe, pad_token=15049):
        self.dataframe = dataframe
        self.pad_token = pad_token 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the tokenized sequences for English and Tamil
        text =  self.dataframe.iloc[idx]["Sentences"]
        source_tokens =  torch.tensor(self.dataframe.iloc[idx]["Padded_source"],  dtype=torch.long) 
        target_tokens = torch.tensor(self.dataframe.iloc[idx]["Padded_target"],  dtype=torch.long)   

        def causal_mask(size):
              mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
              return mask == 0
    # Return the sequence and masks in a dictionary
        return {
            "source": text,
            "source_tokens": source_tokens.clone(),
            "target_tokens": target_tokens.clone(),
            "decoder_mask": (target_tokens != self.pad_token).unsqueeze(0).int() & causal_mask(target_tokens.size(0)).clone(),

        }