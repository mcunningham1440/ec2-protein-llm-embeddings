# Generating protein LLM embeddings using parallelized EC2 compute

This project represents a fully automated pipeline to generate deep embedding vectors for a set of desired proteins using a pretrained protein large language model (LLM)—Meta's ESM-2—on a multi-GPU-equipped AWS EC2 instance. Running the start_instance.sh script automatically performs all the actions necessary to:

1. Launch the appropriate EC2 instance and transfer the necessary files
2. Set up the instance to generate deep embeddings using a model from the HuggingFace transformers libarary
3. Download the ESM-2 model and perform pipeline parallelization across multiple GPUs
4. Generate the embeddings for the full protein dataset
5. Transfer the embeddings back from the EC2 instance to the local machine
6. Terminate the instance

Some considerations users should be aware of:

- The instance used—g5.12xlarge—is expensive, costing $5.67 per hour as of February 2024. Be cognizant of potential costs before using, and manually double-check that the script has successfully terminated the instance
- Ensure that the csv file containing the protein sequences to be embedded follows the pattern of to_instance/proteins.csv exactly so that the script runs properly