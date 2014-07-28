w2v4j - Word2Vec for Java
======
####Overview
w2v4j is a java library for [Google's Word2Vec](http://code.google.com/p/word2vec/), a tool for computing distributed vector representaions for words.

####How to use
To train the word2vec model on a corpus:

```
Word2Vec model = new Word2VecBuilder(PATH_OF_TRAIN_CORPUS).build();
model.train();
```

You can also set parameters for the model, an example for setting the learning rate is as follows:

```
Word2Vec model = new Word2VecBuilder(PATH_OF_TRAIN_CORPUS).alpha(0.1).build();
```

Here are the parameters you can specify for word2vec model:

- alpha: initial learning rate. default value is 0.025
- window: context window size. default value is 5.
- layerSize: dimension of the projection layer, i.e. the dimension of word vector. default value is 100
- minCount: the minimum count frequency. The word below this threshold will not be added to vocabulary. default value is 5
- numWorker: number of threads to train the model. default value is 4

There are other two parameters you can specify:

- sentenceIteratorFactory: The factory that creates the sentenceIterator. The default value is `com.medallia.w2v4j.iterator.LineSentenceIteratorFactory`, which creates `com.medallia.w2v4j.iterator.LineSentenceIterator` given the train corpus file. The default sentence iterator assumes each line of the train corpus file is a sentence. You can customize the behavior by implementing the `com.medallia.w2v4j.SentenceIteratorFactory` and creates corresponding `Iterator<String>`. For example, you can create an iterator that uses a sentence segmentation library.
- tokenizer: Tokenize sentence to words. The default value is `com.medallia.w2v4j.tokenizer.RegexTokenizer` which tokenizes a sentence by whitespace. You can customize the behavior by extending `com.medallia.w2v4j.tokenizer.tokenizer`. For example, you can write a wrapper for [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml), or applying pos tagging/stop word list to filter out some words.


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
model.save(PATH_TO_SAVE);
```

To load model from disk:

```
Word2Vec model = Word2Vec.load(PATH_TO_LOAD);
```