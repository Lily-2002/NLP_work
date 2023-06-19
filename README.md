# ReadMe

This code is a Python implementation of the Skip-Gram model for word embeddings. It utilizes PyTorch and Plotly libraries to train the Skip-Gram model on the PTB (Penn Treebank) dataset and visualize the word embeddings.

The code consists of several functions and classes:

1. `read_ptb`: Reads the PTB dataset from a file and preprocesses it.
2. `subsample`: Subsamples frequent words from the dataset to improve training efficiency.
3. `batchify`: Converts data into batches for efficient training.
4. `get_centers_and_contexts`: Retrieves center words and corresponding contexts from the corpus.
5. `RandomGenerator`: Generates random negative samples for the Skip-Gram model.
6. `get_negatives`: Generates negative samples for all contexts in the dataset.
7. `load_data_ptb`: Loads and preprocesses the PTB dataset for training.
8. `skip_gram`: Implements the Skip-Gram model architecture.
9. `SigmoidBCELoss`: Custom loss function for the Skip-Gram model.
10. `train`: Trains the Skip-Gram model using the PTB dataset.
11. `evaluate`: Evaluates the Skip-Gram model on a validation dataset.
12. `get_similar_tokens`: Retrieves similar tokens to a given query token based on learned embeddings.
13. `reduce_dimensions`: Reduces the dimensionality of the word embeddings for visualization using t-SNE.
14. `plot_with_plotly`: Plots the reduced-dimensional word embeddings using Plotly.

To use this code, the PTB dataset should be provided in a text file format. The code loads the dataset, preprocesses it, trains the Skip-Gram model, and saves the trained model. It also provides functions to retrieve similar tokens and visualize the word embeddings.