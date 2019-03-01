# Examples of using deep learning in Bioinformatics

Deep learning, which is especially formidable in handling big data, has achieved great success in various fields, including bioinformatics. With the advances of the big data era in biology, it is foreseeable that deep learning will become increasingly important in the field and will be incorporated in vast majorities of analysis pipelines.

To facilitate the process, in this repository, we provide eight examples, which cover five research directions, four data types, and a number of deep learning models that people will encounter in Bioinformatics. The five research directions are: sequence analysis, structure prediction and reconstruction, biomolecular property and function prediction, biomedical image processing and diagnosis, biomolecule interaction prediction and systems biology. The four data types are: structured data, 1D sequence data, 2D image or profiling data, graph data. The covered deep learning models are: deep fully connected neural networks, ConvNet, RNN, graph convolutional neural network, ResNet, GAN, VAE.

The related tutorial and review manuscript can be referred here: [Deep learning in bioinformatics: introduction, application, and perspective in big data era](https://www.biorxiv.org/content/10.1101/563601v1)

If you find the tutorial and this repository useful, please cite our manuscript with the following information
```
@article {li2019deep,
	author = {Li, Yu and Huang, Chao and Ding, Lizhong and Li, Zhongxiao and Pan, Yijie and Gao, Xin},
	title = {Deep learning in bioinformatics: introduction, application, and perspective in big data era},
	elocation-id = {563601},
	year = {2019},
	doi = {10.1101/563601},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2019/02/28/563601},
	eprint = {https://www.biorxiv.org/content/early/2019/02/28/563601.full.pdf},
	journal = {bioRxiv}
}

```


Here is the overview of the eight examples:

#### 1.Fully_connected_psepssm_predict_enzyme
This example shows how to use a neural network to identify enzymes.

* Model: deep fully connected neural network
* Data type: structured data
* Research direction: biomolecular property and function prediction

#### 2.CNN_RNN_sequence_analysis
This example shows how to use the combination of CNN and RNN to predict the non-coding DNA sequence function.

* Model: CNN, RNN
* Data type: 1D sequence data
* Research direction: sequence analysis

#### 3.Regression_gene_expression
This example shows how to use deep learning to predict target gene expression with the landmark gene expression data.

* Model: deep fully connected neural network
* Data type: structured data
* Research direction: biomolecule interaction prediction and systems biology

#### 4.ResNet_X-ray_classification
This example shows how to perform diagnosis with ResNet on the X-ray images.

* Model: ResNet
* Data type: 2D image or profiling data
* Research direction: biomedical image processing and diagnosis

#### 5.GNN_PPI_network
This example shows how to using graph neural network to perform graph embedding and predict protein protein interactions in PPI network.

* Model: graph convolutional neural network
* Data type: graph data
* Research direction: biomolecule interaction prediction and systems biology

#### 6.GAN_image_SR
This example shows how to perform biological image super resolution with GAN.

* Model: GAN
* Data type: 2D image or profiling data
* Research direction: biomedical image processing and diagnosis

#### 7.VAE_high_dim_biological_data_embedding_generation
This example shows how to use VAE to reduce the dimensionality of gene expression profile.

* Model: VAE
* Data type: 2D image or profiling data
* Research direction: biomolecule interaction prediction and systems biology

#### 8.RBP_prediction_CNN
This example shows how to perform RNA-protein binding sites prediction with CNN.

* Model: CNN
* Data type: 1D sequence data
* Research direction: sequence analysis

