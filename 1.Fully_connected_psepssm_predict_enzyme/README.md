# 1.Fully_connected_psepssm_predict_enzyme

In the implementation, we use functional domain encoding as feature and deep fully connected neural network as model to identify enzyme sequences from protein sequences. We use ReLU as the activation function, cross-entropy loss as the loss function, and Adam as the optimizer. We utilize dropout, batch normalization and weight decay to prevent overfitting. With the help of Keras, we build and train the model in 10 lines. Training the model on a Titan X for 2 minutes, we can reach around 94.5% accuracy, which is very close to the state-of-the-art performance.


Reference: https://academic.oup.com/bioinformatics/article/34/5/760/4562505