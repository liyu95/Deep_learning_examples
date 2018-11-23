# 8.RBP_prediction_CNN


In this example, we show how to predict the RBP binding site using CNN, with the data from [Predicting RNA-protein binding sites and motifs through combining local and global deep convolutional neural networks](https://www.ncbi.nlm.nih.gov/pubmed/29722865). Specifically, the task is to predict whether a certain RBP, which is fixed for one model, can bind to a certain given RNA sequence, that is, a binary classification problem. We use one-hot encoding to convert the RNA sequence strings of 'AUCG' into 2D tensors. For example, for 'A', we use a vector (1,0,0,0) to represent it; for 'U', we use a vector (0,1,0,0) to represent it. Concatenating those 1D vectors into a 2D tensor in the same order as the original sequence, we obtain the one-hot encoding for a certain RNA sequence. Notice that for the one-hot encoding, we can consider it either as a 2D map with 1 channel or a 1D vector with 4 channels. Correspondingly, for the convolutional layers, we can choose either 2D convolutions or 1D convolutions. In this example, we follow the previous research setting, using 2D convolutions. The original implementation of [Predicting RNA-protein binding sites and motifs through combining local and global deep convolutional neural networks](https://www.ncbi.nlm.nih.gov/pubmed/29722865) is in Pytorch. We reimplemented the idea using Keras, which builds the model in 15 lines. 


Reference:
* [Predicting RNA-protein binding sites and motifs through combining local and global deep convolutional neural networks](https://www.ncbi.nlm.nih.gov/pubmed/29722865), https://github.com/xypan1232/iDeepE



