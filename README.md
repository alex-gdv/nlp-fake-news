# Natural Language Processing with Fake News

In this project, I worked on the [Fake News Challenge](https://www.fakenewschallenge.org/).
First, I explored the data using different visualisation techniques, such as word clouds and histograms. I used Matplotlib and WordCloud in this step. 
Then, I cleaned and prepared the data using pandas, NLTK, and regular expressions. 
I removed stopwords and links and lemmatised each word. Once I had prepared the data, I created two types of numerical embeddings for the text data. 
I used the TF-IDF vectoriser from Scikit-learn to make a vector for each item. I then used Singular Value Decomposition to make the vector smaller and more suitable for a machine learning model. 
I used the BERT language model from Hugging Face for the second embedding. However, the embedded data was too large for the RAM of the university's GPU cluster. 
Therefore, I divided the dataset into smaller subdatasets. Once I had embedded the data, I trained four machine learning models. 
I used the support vector machine from Scikit-learn and a one-dimensional convolutional neural network. Based on research, I created the architecture for the CNN using PyTorch. 
I had to design two separate architectures as the embedded data from TF-IDF had a different shape to the BERT data. 
After training the models on the GPU cluster, I performed various tests to evaluate model performance, collected the results, and produced a scientific report discussing my methodology and findings. 
The one-dimensional CNN trained on the TF-IDF data outperformed the other models and achieved 78% accuracy.
