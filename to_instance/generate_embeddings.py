from transformers import AutoTokenizer, EsmConfig, EsmModel
from accelerate import init_empty_weights, infer_auto_device_map
import torch
import numpy as np
import pandas as pd

model_config = EsmConfig.from_pretrained('facebook/esm2_t33_650M_UR50D')
tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
model.train(False)

proteins = pd.read_csv('proteins.csv')
proteins = proteins[proteins['Sequence'].str.len() <= 1024]

def batch_idx_generator(n_proteins, batch_size):
    for i in range(0, n_proteins, batch_size):
        yield range(i, min(i + batch_size, n_proteins))

embeddings_dict = {}

for batch_indices in batch_idx_generator(len(proteins), 32):
    test_seqs = proteins['Sequence'].iloc[batch_indices].tolist()
    batch_protein_names = proteins['UniProt'].iloc[batch_indices].tolist()
    input = tokenizer(test_seqs, return_tensors='pt', padding=True, truncation=False)

    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
    
    outputs = outputs[:,1:-1,:].mean(1)

    for name, embedding in zip(batch_protein_names, outputs):
        embeddings_dict[name] = embedding.numpy()

    print(f"{batch_indices[-1] + 1} processed of {len(proteins)}")

np.savez('protein_embeddings.npz', **embeddings_dict)