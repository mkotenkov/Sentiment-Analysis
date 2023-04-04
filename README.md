## Maxim Koltiugin

<img src="./preprocessing.png" width="450px"></img>

In this project I've tryed different aprroaches to preprocess text:
- **tf-idf.** Converts each document to a vector of size **5353**. Columns correspond to the tf-idf metric of the most popular words in the dataset, excluding overpopular and stop words.
- **doc2idxs.** This is not an official name of the method. Here I transform each document into a vector of size **200** (max_text_length) which contain indexes of words according to generated vocabulary (containing 36879 words). Futher, they will be used by torch.Embedding() to self-learning embeddings.
- **doc2vec.** Here I'm using Navec's pre-trained embeddings, which are summed up to make up a document embedding.
- **doc2matrix.** Same as mentioned above, but embeddings are not summed. Used for reccurent nets.


**Note**: Content of files **'Data/balanced'** and **'Data/unbalanced'** was removed due to the project size. But it can be restored by running files **Pre-processing_balanced.ipynb** and **Pre-processing_unbalanced.ipynb** respectively.
