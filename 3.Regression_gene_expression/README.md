# 3.Regression_gene_expression

To use this repository, first download the data [here](https://drive.google.com/file/d/1W0_q3KTAF7Yf0AzUThqb3Ic778y7Zdrs/view?usp=sharing) into this folder.

In this example, we use deep learning method to perform gene expression prediction as in [D-GEX](https://academic.oup.com/bioinformatics/article/32/12/1832/1743989), showing how to perform regression using deep learning. We further processed the Gene Expression Omnibus (GEO) dataset from [D-GEX](https://academic.oup.com/bioinformatics/article/32/12/1832/1743989), which has gone through the standard normalization procedure. For the deep learning architecture, we use a deep fully connected neural network. For this regression problem, we use the mean squared error as the loss function. Accordingly, we changed the activation function of the last layer from Softmax to TanH for this application. Using Keras, the network can be built and trained in 10 lines. Trained on a Titan X card for 2 mins, it can outperform the linear regression method by 4.5\% on a randomly selected target gene.

Reference: 
* [Gene expression inference with deep learning](https://academic.oup.com/bioinformatics/article/32/12/1832/1743989), https://github.com/uci-cbcl/D-GEX