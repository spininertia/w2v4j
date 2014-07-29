package com.medallia.w2v4j;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Queues;
import com.medallia.w2v4j.WordVector.Code;
import com.medallia.w2v4j.iterator.IteratorUtils;
import com.medallia.w2v4j.tokenizer.RegexTokenizer;
import com.medallia.w2v4j.tokenizer.Tokenizer;
import com.medallia.w2v4j.utils.MathUtils;
import com.sun.istack.internal.Nullable;

/**
 * {@Code Word2VecTrainer} defines the trainer for word2vec model.
 */
public class Word2VecTrainer {
	private static final Logger logger = LogManager.getRootLogger(); 
	
	static final double MIN_ALPHA = 0.0001; // don't allow learning rate to drop below this threshold
	static final int CHUNK_SIZE = 10000;	// number of sentences per chunk
	static final String OOV_WORD = "<OOV>"; // mark for out-of-vocabulary word
	
	
	final int numWorker;		// number of threads for training
	final int window;			// window size 
	final int layerSize;		// dimension of word vector
	final int minCount; 		// the minimum frequency that a word in vocabulary needs to satisfy
	final double initAlpha;		// initial learning rate

	final Tokenizer tokenizer;
	
	private Word2VecTrainer(Word2VecTrainerBuilder builder) {
		this.window = builder.window;
		this.layerSize = builder.layerSize;
		this.minCount = builder.minCount;
		this.initAlpha = builder.initAlpha;
		this.tokenizer = builder.tokenizer;
		this.numWorker = builder.numWorker;
	}

	/** Builder for Word2Vec. */
	public static class Word2VecTrainerBuilder {
		private int window = 5;
		private int layerSize = 100;
		private int minCount = 5;
		private double initAlpha = 0.025;
		private int numWorker = 4;
		
		private Tokenizer tokenizer;
		
		public Word2VecTrainerBuilder() {
			this.tokenizer = new RegexTokenizer();
		}
		
		public Word2VecTrainerBuilder window(int window) {
			this.window = window;
			return this; 
		}
		
		public Word2VecTrainerBuilder layerSize(int layerSize) {
			this.layerSize = layerSize;
			return this;
		}
		
		public Word2VecTrainerBuilder minCount(int minCount) {
			this.minCount = minCount;
			return this;
		}
		
		public Word2VecTrainerBuilder alpha(double alpha) {
			this.initAlpha = alpha;
			return this;
		}
		
		public Word2VecTrainerBuilder tokenizer(Tokenizer tokenizer) {
			this.tokenizer = tokenizer;
			return this;
		}
		
		public Word2VecTrainerBuilder numWorker(int numWorker) {
			this.numWorker = numWorker;
			return this;
		}
		
		public Word2VecTrainer build() {
			return new Word2VecTrainer(this);
		}
	}
	
	/** Train skip-gram model from file. */
	public Word2VecModel train(File file) {
		return train(IteratorUtils.fileSentenceIterable(file));
	}
	
	/** Train skip-gram model with Hierarchical Softmax. */
	public Word2VecModel train(Iterable<String> sentences) {
		logger.info("Start building model...");
		Word2VecModel model = new Word2VecModel(this);
		buildVocabulary(model, sentences);
		buildModel(model, sentences);
		return model;
	}
	
	private void buildModel(Word2VecModel model, Iterable<String> sentences) {
		ModelLocalVariable variable = new ModelLocalVariable();
		
		ExecutorService executor = Executors.newFixedThreadPool(numWorker);
		ExecutorCompletionService<Integer> executorCompletionService = new ExecutorCompletionService<Integer>(executor);
		Iterable<List<String>> chunks = Iterables.partition(sentences, CHUNK_SIZE);
		int numCurrent = 0;
		int numComplete = 0;

		// submit task only when a thread is idle, this makes streaming in chunks possible
		for (List<String> chunk : chunks) {
			numCurrent++;
			if(numCurrent > numWorker) {
				try {
					executorCompletionService.take();
					numComplete++;
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			} 
			executorCompletionService.submit(new Word2VecWorker(model, variable, chunk));
		}
		
		while (numComplete < numCurrent) {
			try {
				executorCompletionService.take();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			numComplete++;
		}
		
		normalize(model);
	}
	
	private void buildVocabulary(Word2VecModel model, Iterable<String> sentences) {
		Map<String, Long> counter = Maps.newHashMap();
		
		for (String sentence : sentences) {
			String[] words = tokenizer.tokenize(sentence);
			
			for (String word : words) {
				if (counter.containsKey(word)) {
					counter.put(word, counter.get(word) + 1);
				} else {
					counter.put(word, 1l);
				}
			}
		}
		
		HuffmanTree huffman = new HuffmanTree();
		model.vocab = huffman.of(counter);
		model.wordCount = huffman.wordCount;
		
		logger.info("Vocabulary established..");
		logger.info(String.format("numWords in Corpus:%d", model.wordCount));
	}

	private void normalize(Word2VecModel model) {
		for (WordVector wordNueron: model.vocab.values()) {
			MathUtils.normalize(wordNueron.vector);
		}
	}
	
	boolean isOovWord(String word) {
		return OOV_WORD.equals(word);
	}
	
	/** Replace OOV word with a OOV_WORD mark, this prevents repeated hashmap look-up. */
	int replaceOovWords(String[] sentence, Word2VecModel model) {
		int numValidWord = 0; 
		for (int i = 0; i < sentence.length; i++) {
			if (!model.vocab.containsKey(sentence[i])) {
				sentence[i] = OOV_WORD;
			} else {
				numValidWord++;
			}
		}
		return numValidWord;
	}
	
	// variables that needs to synchronize among workers on different threads
	// abstract these parameters out to assert the train method has no side effect and is thread safe
	class ModelLocalVariable {
		AtomicLong wordCount = new AtomicLong(0);		// current word count
		AtomicLong sentenceCount = new AtomicLong(0);	// current sentence count
		volatile double alpha;							// learning rate, decreasing as training proceeds
	}

	/**
	 * HuffmanTree for creating tree strucutre in Hierarchical Softmax. 
	 */
	private class HuffmanTree {
		Map<String, WordVector> vocab = Maps.newHashMap();
		int wordCount = 0;
		
		/** Create Huffman Tree for Hierarchical Softmax. */
		private ImmutableMap<String, WordVector> of(Map<String, Long> wordMap) {
			logger.info("Creating huffman tree...");
			
			Queue<HuffmanNode> heap = Queues.newPriorityQueue();
			for (Entry<String, Long> entry : wordMap.entrySet()) {
				// filter out words below minCount
				if (entry.getValue() >= minCount) {
					wordCount += entry.getValue();
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
			return ImmutableMap.copyOf(vocab);
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
		
		class HuffmanNode implements Comparable<HuffmanNode>, Serializable{
			String word; // nullable, inner node don't have such field. 
			long count;
			HuffmanNode left;
			HuffmanNode right;
			
			public HuffmanNode(@Nullable String word, long count) {
				this.word = word;
				this.count = count;
				left = right = null;
			}
	
			public int compareTo(HuffmanNode node) {
				long diff = this.count - node.count;
				
				if (diff > 0) {
					return 1;
				} else if (diff < 0){ 
					return -1;
				}
				return 0;
			}
			
			@Override
			public String toString() {
				return String.format("%s %d", word == null ? "inner" : word, count);
			}
		}
	}
	
	class Word2VecWorker implements Callable<Integer> {
		Word2VecModel model;
		Iterable<String> sentences;
		ModelLocalVariable variable;
		
		
		public Word2VecWorker(Word2VecModel model, ModelLocalVariable variable, Iterable<String> sentences) {
			this.model = model;
			this.variable = variable;
			this.sentences = sentences;
		}
		
		public Integer call() {
			trainChunk();
			return 0;
		}
		
		private void trainChunk() {
			for (String sentence : sentences) {
				String[] words = tokenizer.tokenize(sentence);
				long crtWordCount = variable.wordCount.addAndGet(replaceOovWords(words, model));
				long sentCount = variable.sentenceCount.incrementAndGet();
				if (sentCount % 100 == 0) {
					variable.alpha = Math.max(Word2VecTrainer.MIN_ALPHA, initAlpha * (1 - 1.0 * crtWordCount / model.wordCount));
					if (sentCount % 10000 == 0) {
						logger.info(String.format("%d sentences trained..", sentCount));
					}
				}
				trainSkipGramForSentence(variable.alpha, words);
			}
		}
		
		private void trainSkipGramForSentence(double alpha, String[] sentence) {
			for (int contextWordPos = 0; contextWordPos < sentence.length; contextWordPos++) {
				String contextWord = sentence[contextWordPos];
				// skip OOV word
				if (isOovWord(contextWord)) continue;
				
				WordVector contextWordNeuron = model.vocab.get(contextWord);
				int reducedWindow = ThreadLocalRandom.current().nextInt(window);
				int start = Math.max(0, contextWordPos - window + reducedWindow);
				int end = Math.min(sentence.length, contextWordPos + window + 1 - reducedWindow);
				
				for (int inputWordPos = start; inputWordPos < end; inputWordPos++) {
					if (inputWordPos == contextWordPos) continue;
					
					String inputWord = sentence[inputWordPos];
					// skip OOV word
					if (isOovWord(inputWord)) continue;
					WordVector inputWordNeuron = model.vocab.get(inputWord);
					
					double[] inputWordVector = Arrays.copyOf(inputWordNeuron.vector, layerSize);
					for (int i = 0; i < contextWordNeuron.getCodeLen(); i++) {
						NodeVector nodeNeuron = contextWordNeuron.points.get(i);
						Code code = contextWordNeuron.code.get(i);
						double prob = MathUtils.sigmoid(MathUtils.dotProduct(nodeNeuron.vector, inputWordVector));
						double gradient = (1 - code.getValue() - prob) * alpha;
						MathUtils.gradientUpdate(inputWordNeuron.vector, nodeNeuron.vector, gradient);
						MathUtils.gradientUpdate(nodeNeuron.vector, inputWordVector, gradient);
					}
				}
			}
		}

	}

}
