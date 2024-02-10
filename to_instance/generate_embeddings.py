from transformers import AutoTokenizer, EsmConfig, EsmModel
from accelerate import init_empty_weights, infer_auto_device_map
import torch
import numpy as np
import pandas as pd

model_path = "facebook/esm2_t48_15B_UR50D" #Path to HuggingFace repo for model
n_gpus = 4
gpu_memory = 24 #GiB of memory per GPU
batch_size = 1 #n of sequences to embed at once
max_seq_length = 1024 #Maximum protein sequence length allowed by model

model_config = EsmConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Allows the correct device mapping to be inferred before loading model weights
with init_empty_weights():
  empty_model = EsmModel(model_config)

device_map = infer_auto_device_map(
    empty_model, 
    max_memory={gpu_id: f"{gpu_memory}GiB" for gpu_id in range(n_gpus)}
    )

model = EsmModel.from_pretrained(model_path, device_map=device_map)
model.train(False)

proteins = pd.read_csv('to_instance/proteins.csv')
proteins = proteins[proteins['Sequence'].str.len() <= max_seq_length]

def batch_idx_generator(n_proteins, batch_size):
    for i in range(0, n_proteins, batch_size):
        yield range(i, min(i + batch_size, n_proteins))

embeddings_dict = {}

for batch_indices in batch_idx_generator(len(proteins), batch_size):
    test_seqs = proteins['Sequence'].iloc[batch_indices].tolist()
    batch_protein_names = proteins['UniProt'].iloc[batch_indices].tolist()
    test_seqs = tokenizer(
        test_seqs, 
        return_tensors='pt', 
        padding=True, 
        truncation=False
        ).to(0)

    # Using the last hidden state of the model as the embedding, mean-aggregated
    # over the sequence length
    with torch.no_grad():
        outputs = model(**test_seqs).last_hidden_state
    outputs = outputs[:,1:-1,:].mean(1)

    # Storing the embeddings in a dictionary and converting to numpy
    for name, embedding in zip(batch_protein_names, outputs):
        embeddings_dict[name] = embedding.to('cpu').numpy()

    print(f"{batch_indices[-1] + 1} of {len(proteins)} generated")

np.savez('protein_embeddings.npz', **embeddings_dict)