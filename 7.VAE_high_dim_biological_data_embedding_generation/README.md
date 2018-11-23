# 7.VAE_high_dim_biological_data_embedding_generation

To use this example, first download the data [here](https://drive.google.com/file/d/1iH6Z_aDA-DL33XEdapyfkzPls3Aji9Hh/view?usp=sharing) and decompress the file into this folder.

In this example, we briefly show how to use VAE to perform dimensionality reduction for gene expression data. We use the dataset from [Extracting a biologically relevant latent space from cancer transcriptomes with variational autoencoders](https://www.ncbi.nlm.nih.gov/pubmed/29218871), preprocessed from the TCGA database, which collects the gene expression data for over 10,000 different tumors. Within the database, the RNA-seq data describe the high-dimensional state of each tumor. In the dataset we use, the dimensionality is 5,000 for each tumor. Using VAE, we are able to reduce the dimensionality to 100. In that space, it is easier for us to identify the common patterns and signatures between different tumors. We used Keras to implement VAE. For the loss, we use mean-squared error.


Reference:
* [Extracting a biologically relevant latent space from cancer transcriptomes with variational autoencoders](https://www.ncbi.nlm.nih.gov/pubmed/29218871), https://github.com/greenelab/tybalt

Further references if you are interested in drug design with VAE:
* https://github.com/samsinai/VAE_protein_function
* https://github.com/rampasek/DrVAE
