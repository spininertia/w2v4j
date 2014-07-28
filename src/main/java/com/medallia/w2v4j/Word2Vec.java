package com.medallia.w2v4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Queues;
import com.medallia.w2v4j.WordVector.Code;
import com.medallia.w2v4j.iterator.LineSentenceIteratorFactory;
import com.medallia.w2v4j.iterator.SentenceIteratorFactory;
import com.medallia.w2v4j.tokenizer.RegexTokenizer;
import com.medallia.w2v4j.tokenizer.Tokenizer;
import com.medallia.w2v4j.utils.FileSplitter;
import com.medallia.w2v4j.utils.Utils;

/**
 * @{link {@link Word2Vec} is the main entry for the w2v4j library. 
 */
public class Word2Vec implements Serializable{
	private static Logger logger = LogManager.getRootLogger(); 
	
	static final double MIN_ALPHA = 0.0001; // don't allow learning rate to drop below this threshold
	static final String OOV_WORD = "<OOV>";
	
	final int numWorker;		// number of threads for training
	final int window;			// window size 
	final int layerSize;		// dimension of word vector
	final int minCount; 		// the minimum frequency that a word in vocabulary needs to satisfy
	final double initAlpha;		// initial learning rate
	
	long totalWords = 0;							// aggregated word frequency
	AtomicLong wordCount = new AtomicLong(0);		// current word count
	AtomicLong sentenceCount = new AtomicLong(0);	// current sentence count
	volatile double alpha;							// learning rate, decreasing as training proceeds

	Map<String, WordVector> vocab = Maps.newHashMap();	// map from word to its WordNeuron
	
	File trainCorpus;
	SentenceIteratorFactory sentenceIteratorFactory;
	Tokenizer tokenizer;
	
	private Word2Vec(Word2VecBuilder builder) {
		this.window = builder.window;
		this.layerSize = builder.layerSize;
		this.minCount = builder.minCount;
		this.initAlpha = builder.startingAlpha;
		this.sentenceIteratorFactory = builder.sentenceIteratorFactory;
		this.tokenizer = builder.tokenizer;
		this.trainCorpus = builder.trainCorpus;
		this.numWorker = builder.numWorker;
	}

	/** Builder for Word2Vec. */
	public static class Word2VecBuilder {
		private int window = 5;
		private int layerSize = 100;
		private int minCount = 5;
		private double startingAlpha = 0.025;
		private int numWorker = 4;
		
		private SentenceIteratorFactory sentenceIteratorFactory;
		private Tokenizer tokenizer;
		private File trainCorpus;
		
		public Word2VecBuilder(File trainCorpus) {
			this.trainCorpus = trainCorpus;
			this.tokenizer = new RegexTokenizer();
			sentenceIteratorFactory = LineSentenceIteratorFactory.getInstance();
		}
		
		public Word2VecBuilder(String trainFilePath) {
			this(new File(trainFilePath));
		}
		
		public Word2VecBuilder window(int window) {
			this.window = window;
			return this; 
		}
		
		public Word2VecBuilder layerSize(int layerSize) {
			this.layerSize = layerSize;
			return this;
		}
		
		public Word2VecBuilder minCount(int minCount) {
			this.minCount = minCount;
			return this;
		}
		
		public Word2VecBuilder alpha(double alpha) {
			this.startingAlpha = alpha;
			return this;
		}
		
		public Word2VecBuilder sentenceIteratorFactory(SentenceIteratorFactory factory) {
			this.sentenceIteratorFactory = factory;
			return this;
		}
		
		public Word2VecBuilder tokenizer(Tokenizer tokenizer) {
			this.tokenizer = tokenizer;
			return this;
		}
		
		public Word2VecBuilder numWorker(int numWorker) {
			this.numWorker = numWorker;
			return this;
		}
		
		public Word2Vec build() {
			return new Word2Vec(this);
		}
	}
	
	/** Serialize and save the {@link Word2Vec} model. */
	public void save(String path) {
		ObjectOutputStream out;
		try {
			out = new ObjectOutputStream(new FileOutputStream(path));
			out.writeObject(this);
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	/** Load and deserialize model from disk. */
	public static Word2Vec load(String path) {
		Word2Vec model = null;
		try {
			ObjectInputStream input = new ObjectInputStream(new FileInputStream(path));
			model = (Word2Vec) input.readObject();
			input.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		return model;
	}
	
	/** Compute cosine Similarity for two words. */
	public double similarity(String word1, String word2) {
		if (word1 == null || word2 == null) {
			return 0;
		}
		
		if (word1.equals(word2)) return 1;
		
		WordVector neuron1 = vocab.get(word1);
		WordVector neuron2 = vocab.get(word2);
		
		if (neuron1 == null || neuron2 == null) {
			return 0;
		}
		
		// this computes cosine similarity since vectors are already normalized
		return Utils.dotProduct(neuron1.vector, neuron2.vector);
	}
	
	/** Get n most similar words to word along with their similarity. */
	public List<WordWithSimilarity> mostSimilar(String word, int n) {
		List<WordWithSimilarity> result = Lists.newArrayList();
		
		if (!vocab.containsKey(word)) {
			return result;
		}
		for (Entry<String, WordVector> entry : vocab.entrySet()) {
			String word2 = entry.getKey();
			if (!word2.equals(word)) {
				result.add(new WordWithSimilarity(word2, similarity(word, word2)));
			}
		}
		Collections.sort(result);
		return result.subList(0, n);
	}
	
	/** Determine whether word is in vocabulary. */
	public boolean containsWord(String word) {
		return vocab.containsKey(word);
	}
	
	/** Returns the defensive copy of the word vector */
	public double[] getWordVector(String word) {
		if (!containsWord(word)) {
			return null;
		}
		return Arrays.copyOf(vocab.get(word).vector, layerSize);
	}
	
	/** Train skip-gram model with Hierarchical Softmax. */
	public void train() {
		logger.info("Start building model...");
		buildVocabulary();
		logger.info("Vocabulary established..");
		buildModel();
	}
	
	private void buildModel() {
		List<String> chunks = null;
		try {
			chunks = FileSplitter.chunkByLine(trainCorpus.getAbsolutePath(), numWorker);
		} catch (IOException e) {
			logger.warn("Chunk Faield");
			return;
		}
		
		List<Thread> threads = Lists.newArrayList();
		for (int i = 0; i < numWorker; i++) {
			Thread thread = new Thread(new Word2VecWorker(i, this, chunks.get(i)));
			threads.add(thread);
			thread.start();
			logger.info("thread " + i + " started!");
		}
		
		for (Thread thread : threads) {
			try {
				thread.join();
			} catch (InterruptedException e) {
				logger.warn("Thread " + thread.toString() + " interupted" );
			}
		}
		
		logger.info("All threads completed..");
		
		normalize();
	}

	private void normalize() {
		for (WordVector wordNueron: vocab.values()) {
			Utils.normalize(wordNueron.vector);
		}
	}
	
	boolean isOovWord(String word) {
		return OOV_WORD.equals(word);
	}
	
	/** Replace OOV word with a OOV_WORD mark, this prevents repeated hashmap look-up. */
	int replaceOovWords(String[] sentence) {
		int numValidWord = 0; 
		for (int i = 0; i < sentence.length; i++) {
			if (!vocab.containsKey(sentence[i])) {
				sentence[i] = OOV_WORD;
			} else {
				numValidWord++;
			}
		}
		return numValidWord;
	}
	
	private void buildVocabulary() {
		Map<String, Long> counter = Maps.newHashMap();
		Iterator<String> sentenceIterator = sentenceIteratorFactory.createSentenceIterator(trainCorpus);
		
		while (sentenceIterator.hasNext()) {
			String sentence = sentenceIterator.next();
			String[] words = tokenizer.tokenize(sentence);
			
			for (String word : words) {
				if (counter.containsKey(word)) {
					counter.put(word, counter.get(word) + 1);
				} else {
					counter.put(word, 1l);
				}
			}
		}
		
		createHuffmanTree(counter);
		logger.info(String.format("numWords in Corpus:%d", totalWords));
	}
	

	/** Create Huffman Tree for Hierarchical Softmax. */
	private void createHuffmanTree(Map<String, Long> wordMap) {
		logger.info("Creating huffman tree...");
		
		Queue<HuffmanNode> heap = Queues.newPriorityQueue();
		for (Entry<String, Long> entry : wordMap.entrySet()) {
			// filter out words below minCount
			if (entry.getValue() >= minCount) {
				totalWords += entry.getValue();
				heap.offer(new HuffmanNode(entry.getKey(), entry.getValue()));
			}
		}
		
		int vocabSize = heap.size();
		for (int i = 0; i < vocabSize - 1; i++) {
			HuffmanNode min1 = heap.poll();
			HuffmanNode min2 = heap.poll();
			
			HuffmanNode innerNode = new HuffmanNode(null, min1.count + min2.count);
			innerNode.left = min1;
			innerNode.right = min2;
			heap.offer(innerNode);
		}
		
		dfs(heap.poll(), new ArrayList<NodeVector>(), new ArrayList<Code>());
	}
	
	/** Traverse huffman tree to attach code and points to word neuron */
	private void dfs(HuffmanNode node, List<NodeVector> path, List<Code> code) {
		if (node == null) return;
		if (node.left == null && node.right == null) {
			// leaf node
			vocab.put(node.word, new WordVector(layerSize, Lists.newArrayList(path), Lists.newArrayList(code)));
			return;
		} 
		
		// inner node
		NodeVector nodeNeuron = new NodeVector(layerSize);
		path.add(nodeNeuron);
		
		if (node.left != null) {
			code.add(Code.LEFT);
			dfs(node.left, path, code);
			code.remove(code.size() - 1);
		}
		
		if (node.right != null) {
			code.add(Code.RIGHT);
			dfs(node.right, path, code);
			code.remove(code.size() - 1);
		}
		
		path.remove(path.size() - 1);
	}
}
