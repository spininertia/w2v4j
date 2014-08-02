w2v4j - Word2Vec for Java
======
####Overview
w2v4j is a java library for [Google's Word2Vec](http://code.google.com/p/word2vec/), a tool for computing distributed vector representaions for words.

####How to use
The major class to access this library is `com.medallia.w2v4j.Word2VecModel` and `com.medallia.w2v4j.Word2VecTrainer`. The `Word2VecTrainer` is used to set hyperparameters for training a word2vec model. The `Word2VecModel` is then used to access the trained word representation.


__To train the word2vec model on a corpus__:

```
Word2VecTrainer trainer = new Word2VecBuilder().build();
Word2VecModel model = trainer.train(new File(PATH_TRAIN));
```
The triner assumes each line is a sentence in the corpus file.

You can also pass `Iterable<String>` as argument to trainer's train method by
```
List<String> sentences = ImmutableList.of("this is a word2vec library in java", "It is awesome");
Word2VecModel model = trainer.train(sentences);
```
This allows you to define customized iterator to iterate through different format of file or drectory.

The trainer's train method has no side effect, which means once you build a trainer, you can use this instance to build multiple models on different corpus with same configuration.


You can also set parameters while building the trainer, an example for setting the learning rate is as follows:
```
Word2VecTrainer trainer  = new Word2VecBuilder(PATH_OF_TRAIN_CORPUS).alpha(0.1).build();
```

Here are the parameters you can specify for word2vec model:

- alpha: initial learning rate. default value is 0.025
- window: context window size. default value is 5.
- layerSize: dimension of the projection layer, i.e. the dimension of word vector. default value is 100
- minCount: the minimum count frequency. The word below this threshold will not be added to vocabulary. default value is 5
- numWorker: number of threads to train the model. default value is 4
- sampling: apply sub-sampling to words if true. default value is false
- samplingThreshold: only available when sampling is turned on. default and suggested value is 1e-5


There are other two parameters you can specify:

- tokenizer: Tokenize sentence to words. The default value is `com.medallia.w2v4j.tokenizer.RegexTokenizer` which tokenizes a sentence by whitespace. You can customize the behavior by extending `com.medallia.w2v4j.tokenizer.tokenizer`. For example, you can write a wrapper for [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml), or applying pos tagging/stop word list to filter out some words.
- model: the neural network language model used to train the word representation. default value is NeuralNetworkLanguageModel.SKIP_GRAM (skip gram), another option is NeuralNetworkLanguagemodel.CBOW (continuous bag of words).


__After the model is trained, you can test its functionality using following method__.

To see whether a word appears in vocabulary:

```
boolean exist = model.containsWord(word);
```

To get the similarity between two words:

```
double similarity = model.similarity(word1, word2);
```

To get the most similar words along with their similarity

```
List<WordWithSimilarity> wordList = model.mostSimilar(word);
for (WordWithSimilarity similarWord : wordList)  {
	System.out.println(similarWord.getWord() + "" + similarWord.getSimilarity());
}
```

Or if you just want to get the word vector:

```
double[] vector = model.getWordVector(word);
```


__The library also supports persisting and load the trained model to and from disk.__

To serialize and save the computed model:

```
SerializationUtils.save(model, PATH_TO_SAVE);
```

To load model from disk:

```
Word2VecModel model = SerializationUtils.load(model, PATH_TO_LOAD);
```

Check out w2v4j-example for more example.

#### Paper referenced
[1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf). In Proceedings of Workshop at ICLR, 2013.

[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. [Distributed Representations of Words and Phrases and their Compositionality.](http://arxiv.org/pdf/1310.4546.pdf) In Proceedings of NIPS, 2013.

[3] Bengio, Yoshua and Ducharme, Rejean and Vincent, Pascal. [A Neural probabilistic language models](http://machinelearning.wustl.edu/mlpapers/paper_files/BengioDVJ03.pdf). In Journal of Machine Learning Research 3, 1137-1155. 2003

[4] Morin, F., & Bengio, Y. [Hierarchical Probabilistic Neural Network Language Model](http://www.iro.umontreal.ca/labs/neuro/pointeurs/hierarchical-nnlm-aistats05.pdf).

[5] Mnih, A., & Hinton, G. E. (2009). [A scalable hierarchical distributed language model](http://papers.nips.cc/paper/3583-a-scalable-hierarchical-distributed-language-model.pdf). In Advances in neural information processing systems (pp. 1081-1088).