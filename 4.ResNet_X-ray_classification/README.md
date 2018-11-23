# 4.ResNet_X-ray_classification

To use this example, first download the data from [all data](https://data.mendeley.com/datasets/rscbjbr9sj/3) and uncompress the X-ray data into this folder. Alternatively, you can download the data [here](https://drive.google.com/file/d/1Y9iTkRrfh_2UfoG9o8CRjZc_3rj73nap/view?usp=sharing), for your convenience.


In this example, we use the chest X-ray dataset from [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub). We use Keras to implement this example. We first load the ResNet trained on ImageNet, and then freeze all the layer's weights, except for the last 4 layers'. We substitute the last layer with a new layer containing 2 nodes, since that dataset has two classes. Besides, we resize the chest X-ray images to make them the same dimensionality as the original ResNet input image using bilinear interpolation. Finally, we run the standard optimization procedure with cross-entropy loss and Adam optimizer. In order to prevent overfitting, like what we have done in the previous examples, we use dropout combined with batch normalization and weight decay.

Reference:
* [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub)
* [Data](https://data.mendeley.com/datasets/rscbjbr9sj/3)