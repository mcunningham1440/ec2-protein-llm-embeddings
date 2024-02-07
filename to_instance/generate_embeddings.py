from transformers import AutoTokenizer, EsmConfig, EsmModel
from accelerate import init_empty_weights, infer_auto_device_map
import torch
import numpy as np
import pandas as pd

model_path = "facebook/esm2_t6_8M_UR50D"

model_config = EsmConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = EsmModel.from_pretrained(model_path)
model.to(0)
model.train(False)

proteins = pd.read_csv('to_instance/proteins.csv')
proteins = proteins[proteins['Sequence'].str.len() <= 1024]

def batch_idx_generator(n_proteins, batch_size):
    for i in range(0, n_proteins, batch_size):
        yield range(i, min(i + batch_size, n_proteins))

embeddings_dict = {}

batch_size = 32

for batch_indices in batch_idx_generator(len(proteins), batch_size):
    test_seqs = proteins['Sequence'].iloc[batch_indices].tolist()
    batch_protein_names = proteins['UniProt'].iloc[batch_indices].tolist()
    input = tokenizer(test_seqs, return_tensors='pt', padding=True, truncation=False)
    input.to(0)

    with torch.no_grad():
        outputs = model(**input).last_hidden_state
    
    outputs = outputs[:,1:-1,:].mean(1)

    for name, embedding in zip(batch_protein_names, outputs):
        embeddings_dict[name] = embedding.cpu().numpy()

    print(f"{batch_indices[-1] + 1} of {len(proteins)} generated")

np.savez('protein_embeddings.npz', **embeddings_dict)