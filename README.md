# Generating protein LLM embeddings using EC2 compute

This is a project to create a fully automated pipeline to generate deep embedding vectors for a set of desired proteins using a pretrained protein large language model (LLM)—Meta's ESM-2—on a multi-GPU-equipped AWS EC2 instance. When complete, this repo will enable the following process to be initiated with a single command:

1. Launching the appropriate EC2 instance and transferring the necessary files
2. Setting up the instance to perform inference using ESM-2
3. Downloading the ESM-2 model and performing pipeline parallelization across multiple GPUs
4. Generating the embeddings for the full protein dataset
5. Transferring the embeddings back from the EC2 instance to the local machine
6. Terminating the instance
