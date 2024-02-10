echo "Installing requirements..."
pip install -r to_instance/requirements.txt

echo "Launching embedding script..."
python to_instance/generate_embeddings.py