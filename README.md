w2v4j - Word2Vec for Java
======
####Overview
w2v4j is a java library for [Google's Word2Vec](http://code.google.com/p/word2vec/), a tool for computing distributed vector representaions for words.

####How to use
The major class to access this library is `com.medallia.w2v4j.Word2VecModel` and `com.medallia.w2v4j.Word2VecTrainer`. The `Word2VecTrainer` is used to set hyperparameters for training a word2vec model. The `Word2VecModel` is then used to access the trained word representation.

To train the word2vec model on a corpus:

```
Word2VecTrainer trainer = new Word2VecBuilder().build();
Word2VecModel model = trainer.train(new File(PATH_TRAIN));
```

You can also pass `Iterable<String>` as argument to trainer's train method
```
List<String> sentences = new ArrayList<String>(new String[]{"this is a word2vec library in java", "It is awesome"});
Word2VecModel model = trainer.train(sentences);
```

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


After the model is trained, you can test its functionality using following method.

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


The library also supports persisting and load the trained model to and from disk.

To serialize and save the computed model:

```
SerializationUtils.save(model, PATH_TO_SAVE);
```

To load model from disk:

```
Word2VecModel model = SerializationUtils.load(model, PATH_TO_LOAD);
```

Check out w2v4j-example for more example.