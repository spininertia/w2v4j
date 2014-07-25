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
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Queues;
import com.medallia.w2v4j.WordNeuron.Code;
import com.medallia.w2v4j.iterator.LineSentenceIteratorFactory;
import com.medallia.w2v4j.iterator.SentenceIteratorFactory;
import com.medallia.w2v4j.tokenizer.RegexTokenizer;
import com.medallia.w2v4j.tokenizer.Tokenizer;
import com.medallia.w2v4j.utils.Utils;

/**
 * @{link {@link Word2Vec} is the main entry for the w2v4j library. 
 */
public class Word2Vec implements Serializable{
	private static Logger logger = LogManager.getRootLogger(); 
	
	static final double MIN_ALPHA = 0.0001; // don't allow learning rate to drop below this threshold
	static final String OOV_WORD = "<OOV>";
	
	int window = 5;
	int layerSize = 100;
	int minCount = 100;
	double startingAlpha = 0.025;
	
	long totalWords = 0;
	Random random = new Random();
	
	Map<String, WordNeuron> vocab = Maps.newHashMap();
	
	File trainCorpus;
	SentenceIteratorFactory sentenceIteratorFactory;
	Tokenizer tokenizer;
	
	private Word2Vec(Word2VecBuilder builder) {
		this.window = builder.window;
		this.layerSize = builder.layerSize;
		this.minCount = builder.minCount;
		this.startingAlpha = builder.startingAlpha;
		this.sentenceIteratorFactory = builder.sentenceIteratorFactory;
		this.tokenizer = builder.tokenizer;
		this.trainCorpus = builder.trainCorpus;
	}


	public static class Word2VecBuilder {
		private int window = 5;
		private int layerSize = 100;
		private int minCount = 5;
		private double startingAlpha = 0.024;
		
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
		
		public Word2Vec build() {
			return new Word2Vec(this);
		}
	}
	
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
		
		WordNeuron neuron1 = vocab.get(word1);
		WordNeuron neuron2 = vocab.get(word2);
		
		if (neuron1 == null || neuron2 == null) {
			return 0;
		}
		
		// this computes cosine similarity since vectors are already normalized
		return Utils.dotProduct(neuron1.vector, neuron2.vector);
	}
	
	/** Get most similar words to word along with their similarity */
	public List<WordWithSimilarity> mostSimilar(String word, int topn) {
		List<WordWithSimilarity> result = Lists.newArrayList();
		
		if (!vocab.containsKey(word)) {
			return result;
		}
		for (Entry<String, WordNeuron> entry : vocab.entrySet()) {
			String word2 = entry.getKey();
			if (!word2.equals(word)) {
				result.add(new WordWithSimilarity(word2, similarity(word, word2)));
			}
		}
		Collections.sort(result);
		return result.subList(0, topn);
	}
	
	public boolean isInVocab(String word) {
		return vocab.containsKey(word);
	}
	
	
	public void train() {
		logger.info("Start building model...");
		buildVocabulary();
		logger.info("Vocabulary established..");
		buildModel();
	}
	
	private void buildModel() {
		Iterator<String> sentenceIterator = sentenceIteratorFactory.createSentenceIterator(trainCorpus);
		int sentenceCount = 0;
		int actualWordCount = 0;
		double alpha = startingAlpha;
		
		while (sentenceIterator.hasNext()) {
			String[] words = tokenizer.tokenize(sentenceIterator.next());
			actualWordCount += replaceOovWords(words);
			sentenceCount++;
			
			// update alpha, the implementation here is the same as gensim's version, slightly different from google's original word2vec
			if (sentenceCount % 100 == 0) {
				alpha = Math.max(MIN_ALPHA, startingAlpha * (1 - 1.0 * actualWordCount / totalWords));
			}
			
			if (sentenceCount % 10000 == 0) {
				logger.info(sentenceCount + " sentences trained.." );
			}
			
			trainSkipGramForSentence(alpha, words);
		}
		
		normalize();
	}

	private void trainSkipGramForSentence(double alpha, String[] sentence) {
		for (int contextWordPos = 0; contextWordPos < sentence.length; contextWordPos++) {
			String contextWord = sentence[contextWordPos];
			// skip OOV word
			if (isOovWord(contextWord)) continue;
			
			WordNeuron contextWordNeuron = vocab.get(contextWord);
			int reducedWindow = random.nextInt(window);
			int start = Math.max(0, contextWordPos - window + reducedWindow);
			int end = Math.min(sentence.length, contextWordPos + window + 1 - reducedWindow);
			
			for (int inputWordPos = start; inputWordPos < end; inputWordPos++) {
				if (inputWordPos == contextWordPos) continue;
				
				String inputWord = sentence[inputWordPos];
				// skip OOV word
				if (isOovWord(inputWord)) continue;
				WordNeuron inputWordNeuron = vocab.get(inputWord);
				
				double[] inputWordVector = Arrays.copyOf(inputWordNeuron.vector, layerSize);
				for (int i = 0; i < contextWordNeuron.getCodeLen(); i++) {
					NodeNeuron nodeNeuron = contextWordNeuron.points.get(i);
					Code code = contextWordNeuron.code.get(i);
					double prob = 1.0 / (1 + Math.exp(-Utils.dotProduct(nodeNeuron.vector, inputWordVector))); // TODO change exp to table lookup to speed-up
					double gradient = (1 - code.getValue() - prob) * alpha;
					Utils.gradientUpdate(inputWordNeuron.vector, nodeNeuron.vector, gradient);
					Utils.gradientUpdate(nodeNeuron.vector, inputWordVector, gradient);
				}
			}
		}
	}
	
	private void normalize() {
		for (WordNeuron wordNueron: vocab.values()) {
			Utils.normalize(wordNueron.vector);
		}
	}
	
	private boolean isOovWord(String word) {
		return OOV_WORD.equals(word);
	}
	
	/** Replace OOV word with a OOV_WORD mark, this prevents repeated hashmap look-up. */
	private int replaceOovWords(String[] sentence) {
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
		
		while (!heap.isEmpty()) {
			HuffmanNode min1 = heap.poll();
			HuffmanNode min2 = heap.poll();
			
			// TODO might need to throw exception here if the corpus only have one word
			HuffmanNode innerNode = new HuffmanNode(null, min1.count + min2.count);
			innerNode.left = min1;
			innerNode.right = min2;
			heap.offer(innerNode);
			
			if (heap.size() == 1) {
				break;
			}
		}
		
		// dfs to attach code and points to word neuron.
		dfs(heap.poll(), new ArrayList<NodeNeuron>(), new ArrayList<Code>());
	}
	
	private void dfs(HuffmanNode node, List<NodeNeuron> path, List<Code> code) {
		if (node == null) return;
		if (node.left == null && node.right == null) {
			// leaf node
			vocab.put(node.word, new WordNeuron(layerSize, Lists.newArrayList(path), Lists.newArrayList(code)));
			return;
		} 
		
		// inner node
		NodeNeuron nodeNeuron = new NodeNeuron(layerSize);
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
