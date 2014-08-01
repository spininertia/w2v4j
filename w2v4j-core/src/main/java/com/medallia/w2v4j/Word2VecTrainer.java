package com.medallia.w2v4j;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Queues;
import com.medallia.w2v4j.WordVector.Code;
import com.medallia.w2v4j.iterator.IteratorUtils;
import com.medallia.w2v4j.tokenizer.RegexTokenizer;
import com.medallia.w2v4j.tokenizer.Tokenizer;
import com.medallia.w2v4j.utils.MathUtils;

/**
 * {@link Word2VecTrainer} defines the trainer for word2vec model.
 * <p>
 * Instances of {@link Word2VecTrainer} should be created using the {@link {@link Word2VecTrainerBuilder}
 * and can be re-used to train multiple models
 */
public class Word2VecTrainer {
	private static final Logger logger = LogManager.getLogger(); 
	
	/** Only log a debug message every 10,000 sentences */
	private static final int DEBUG_THROTTLE_FACTOR = 10_000;
	
	private static final double MIN_ALPHA = 0.0001; 				// don't allow learning rate to drop below this threshold
	private static final int BATCH_SIZE = 10_000;					// number of sentences per batch for multi-threaded processing
	private static final String OOV_WORD = "<OOV>"; 				// constant for out-of-vocabulary words
	
	private final int numWorker;							// number of threads for training
	private final int window;								// window size 
	private final int layerSize;							// dimension of word vector
	private final int minCount; 							// the minimum frequency that a word in vocabulary needs to satisfy
	private final double initAlpha;							// initial learning rate
	private final NeuralNetworkLanguageModel model;			// neural network model
	private final boolean sampling;							// enable sub-sampling for frequent words
	private final double samplingThreshold; 				// only useful when sampling is turned on
	
	private final Tokenizer tokenizer;						// tokenizer to split sentences into words
	
	private Word2VecTrainer(Word2VecTrainerBuilder builder) {
		this.window = builder.window;
		this.layerSize = builder.layerSize;
		this.minCount = builder.minCount;
		this.initAlpha = builder.initAlpha;
		this.tokenizer = builder.tokenizer;
		this.numWorker = builder.numWorker;
		this.model = builder.model;
		this.sampling = builder.sampling;
		this.samplingThreshold = builder.samplingThreshold;
	}

	/** Builder for {@link Word2VecTrainer} */
	public static class Word2VecTrainerBuilder {
		private int window = 5;
		private int layerSize = 100;
		private int minCount = 5;
		private double initAlpha = 0.025;
		private int numWorker = Runtime.getRuntime().availableProcessors();
		private boolean sampling = false;
		private NeuralNetworkLanguageModel model = NeuralNetworkLanguageModel.SKIP_GRAM;
		private double samplingThreshold = 1e-5;
		
		private Tokenizer tokenizer;
		
		/**
		 * Create a new {@link Word2VecTrainerBuilder}
		 */
		public Word2VecTrainerBuilder() {
			this.tokenizer = new RegexTokenizer();
		}
		
		/**
		 * Window size, defaults to 5
		 */
		public Word2VecTrainerBuilder window(int window) {
			this.window = window;
			return this; 
		}
		
		/**
		 * Dimension of the word vector, defaults to 100
		 */
		public Word2VecTrainerBuilder layerSize(int layerSize) {
			this.layerSize = layerSize;
			return this;
		}
		
		/**
		 * Minimum frequency of words in the corpus to be considered, defaults to 5
		 */
		public Word2VecTrainerBuilder minCount(int minCount) {
			this.minCount = minCount;
			return this;
		}
		
		/**
		 * Initial learning rate, typically between 0.0 and 1.0
		 * Note the default value is 0.025
		 */
		public Word2VecTrainerBuilder alpha(double alpha) {
			this.initAlpha = alpha;
			return this;
		}
		
		/**
		 * Specify a custom tokenizer, defaults to split on whitespace
		 */
		public Word2VecTrainerBuilder tokenizer(Tokenizer tokenizer) {
			this.tokenizer = tokenizer;
			return this;
		}
		
		/**
		 * Number of concurrent threads to use for training, defaults to {@link Runtime#availableProcessors()}
		 */
		public Word2VecTrainerBuilder numWorker(int numWorker) {
			this.numWorker = numWorker;
			return this;
		}
		
		/** @see {@link NeuralNetworkLanguageModel} */
		public Word2VecTrainerBuilder model(NeuralNetworkLanguageModel model) {
			this.model = model;
			return this;
		}
		
		/** Whether to use subsampling */
		public Word2VecTrainerBuilder sampling(boolean sampling) {
			this.sampling = sampling;
			return this;
		}
		
		/** Threshold to use for sampling, defaults to 1e-5 */
		public Word2VecTrainerBuilder samplingThreshold(double threshold) {
			this.samplingThreshold = threshold;
			return this;
		}
		
		/** @return {@link Word2VecTrainer} object */
		public Word2VecTrainer build() {
			return new Word2VecTrainer(this);
		}
	}

	/** Train model on file input. */
	public Word2VecModel train(File file) {
		return train(IteratorUtils.fileSentenceIterable(file));
	}
	
	/** Train model on a {@link Iterable} of sentences */
	public Word2VecModel train(Iterable<String> sentences) {
		logger.info("Start building model...");
		TrainingContext context = gatherContext(sentences);
		return buildModel(context, sentences);
	}
	
	/**
	 * Builds vocabulary and accumulates word counts
	 */
	private TrainingContext gatherContext(Iterable<String> sentences) {
		HuffmanTree huffman = buildVocabulary(sentences);
		ImmutableMap<String, WordVector> vocab = huffman.vocab;
		long wordCount = huffman.wordCount;
		
		logger.info("Vocabulary established..");
		logger.info(String.format("Number of words in Corpus:%d", wordCount));
		
		return new TrainingContext(vocab, wordCount, window, layerSize);
	}

	/** Creates Huffman tree to build codes for each word */
	private HuffmanTree buildVocabulary(Iterable<String> sentences) {
		Map<String, Long> wordCounts = Maps.newHashMap();
		
		for (String sentence : sentences) {
			String[] words = tokenizer.tokenize(sentence);
			
			for (String word : words) {
				if (wordCounts.containsKey(word)) {
					wordCounts.put(word, wordCounts.get(word) + 1);
				} else {
					wordCounts.put(word, 1l);
				}
			}
		}
		
		return new HuffmanTree(wordCounts);
	}

	/** Train model concurrently */
	private Word2VecModel buildModel(TrainingContext context, Iterable<String> sentences) {
		ExecutorService executor = Executors.newFixedThreadPool(numWorker);
		ExecutorCompletionService<Integer> executorCompletionService = new ExecutorCompletionService<Integer>(executor);
		Iterable<List<String>> chunks = Iterables.partition(sentences, BATCH_SIZE);
		int numCurrent = 0;
		int numComplete = 0;

		// Avoid materializing all sentences in memory at once
		for (List<String> chunk : chunks) {
			numCurrent++;
			if(numCurrent > numWorker) {
				try {
					executorCompletionService.take();
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					throw new IllegalStateException("Interrupted while processing batch", e);
				}
				numComplete++;
			} 
			executorCompletionService.submit(new Word2VecWorker(context, chunk), 0);
		}
		
		// Wait for pending tasks to complete
		while (numComplete < numCurrent) {
			try {
				executorCompletionService.take();
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				throw new IllegalStateException("Interrupted while processing batch", e);
			}
			numComplete++;
		}
		
		// Normalize the values
		for (WordVector wordVector : context.vocab.values()) {
			MathUtils.normalize(wordVector.vector);
		}
		
		return new Word2VecModel(context.vocab);
	}
	
	/** 
	 * @return if the given word is the out of vocabulary constant
	 * Note that we use a reference equality check since we store the constant in the string arrays
	 */
	private static boolean isOovWord(String word) {
		return OOV_WORD == word;
	}

	/**
	 * Wrapper class for thread-safe variables that need to be synchronized among workers on different threads
	 * and constants that are needed for training
	 */ 
	static class TrainingContext {
		// Fixed parameters for training
		private final long totalWordCount;
		private final ImmutableMap<String, WordVector> vocab;
		private final int window;
		private final int layerSize;
		
		// These will be updated by the workers
		private final AtomicLong wordCount = new AtomicLong(0);			// current word count
		private final AtomicLong sentenceCount = new AtomicLong(0);		// current sentence count
		private volatile double alpha;									// learning rate, decreasing as training proceeds
		
		private TrainingContext(ImmutableMap<String, WordVector> vocab, long wordCount, int window, int layerSize) {
			this.vocab = vocab;
			this.totalWordCount = wordCount;
			this.window = window;
			this.layerSize = layerSize;
		}
	}

	/**
	 * HuffmanTree for creating tree structure using Hierarchical Softmax
	 */
	private class HuffmanTree {
		private final ImmutableMap<String, WordVector> vocab;
		private final long wordCount;
		
		/** Create Huffman Tree for Hierarchical Softmax. */
		public HuffmanTree(Map<String, Long> wordMap) {
			// Create a heap where the nodes correspond to (word, count) for words which are frequent enough
			logger.info("Filtering infrequent words...");
			Queue<HuffmanNode> heap = Queues.newPriorityQueue();
			long wordCount = 0;
			for (Entry<String, Long> entry : wordMap.entrySet()) {
				// filter out words below minCount
				if (entry.getValue() >= minCount) {
					wordCount += entry.getValue();
					heap.offer(new HuffmanNode(entry.getKey(), entry.getValue()));
				}
			}
			this.wordCount = wordCount;
			
			logger.info("Creating huffman tree...");
			int vocabSize = heap.size();
			for (int i = 0; i < vocabSize - 1; i++) {
				HuffmanNode min1 = heap.poll();
				HuffmanNode min2 = heap.poll();
				
				HuffmanNode innerNode = new HuffmanNode(null, min1.count + min2.count, min1, min2);
				heap.offer(innerNode);
			}
			
			// Perform dfs storing the code and path for each leaf node during the traversal
			logger.info("Traversing huffman tree to generate codes...");
			Builder<String, WordVector> accumulator = ImmutableMap.builder();
			dfs(heap.poll(), Lists.<NodeVector>newArrayList(), Lists.<Code>newArrayList(), accumulator);
			this.vocab = accumulator.build();
		}
		
		/** Traverse huffman tree to attach code and points to word vector */
		private void dfs(HuffmanNode node, List<NodeVector> path, List<Code> code, Builder<String, WordVector> accumulator) {
			if (node == null)
				return;
			
			// Store word vector for each word into the vocab
			if (node.left == null && node.right == null) {
				accumulator.put(node.word, new WordVector(
						layerSize,
						samplingThreshold,
						// percentage word frequency in the corpus
						1.0 * node.count / wordCount,
						Lists.newArrayList(path),
						Lists.newArrayList(code)
					));
				return;
			} 
			
			// Inner node
			NodeVector nodeVector = new NodeVector(layerSize);
			path.add(nodeVector);
			
			// Recurse left
			if (node.left != null) {
				code.add(Code.LEFT);
				dfs(node.left, path, code, accumulator);
				code.remove(code.size() - 1);
			}
			
			// Recurse right
			if (node.right != null) {
				code.add(Code.RIGHT);
				dfs(node.right, path, code, accumulator);
				code.remove(code.size() - 1);
			}
			
			path.remove(path.size() - 1);
		}
		
		
	}
	
	/** Node of a Huffman tree */
	private static class HuffmanNode implements Comparable<HuffmanNode> {
		private final String word; 
		private final long count;
		HuffmanNode left;
		HuffmanNode right;
		
		/** Creates a leaf {@link HuffmanNode} */
		private HuffmanNode(String word, long count) {
			this(word, count, null, null);
		}
		
		/** Creates an intermediate {@link HuffmanNode} */
		private HuffmanNode(String word, long count, HuffmanNode left, HuffmanNode right) {
			this.word = word;
			this.count = count;
			this.left = left;
			this.right = right;
		}

		@Override public int compareTo(HuffmanNode node) {
			long diff = this.count - node.count;
			
			if (diff > 0) {
				return 1;
			} else if (diff < 0){ 
				return -1;
			}
			return 0;
		}
		
		@Override public String toString() {
			return String.format("%s %d", word == null ? "inner" : word, count);
		}
	}
	
	/** Type of neural network language model */
	public enum NeuralNetworkLanguageModel {
		/** Skip-Gram Model */
		SKIP_GRAM {
			@Override
			public void trainOnSentence(String[] sentence, TrainingContext context) {
				for (int contextWordPos = 0; contextWordPos < sentence.length; contextWordPos++) {
					String contextWord = sentence[contextWordPos];
					// skip OOV word
					if (isOovWord(contextWord))
						continue; 
					
					WordVector contextWordVector = context.vocab.get(contextWord);
					int reducedWindow = ThreadLocalRandom.current().nextInt(context.window);
					int start = Math.max(0, contextWordPos - context.window + reducedWindow);
					int end = Math.min(sentence.length, contextWordPos + context.window + 1 - reducedWindow);
					
					for (int inputWordPos = start; inputWordPos < end; inputWordPos++) {
						if (inputWordPos == contextWordPos)
							continue;
						
						String inputWord = sentence[inputWordPos];
						// skip OOV word
						if (isOovWord(inputWord))
							continue; 
						
						WordVector inputWordVector = context.vocab.get(inputWord);
						double[] inputWordVectorCopy = Arrays.copyOf(inputWordVector.vector, context.layerSize);
						
						for (int i = 0; i < contextWordVector.getCodeLen(); i++) {
							NodeVector nodeVector = contextWordVector.points.get(i);
							Code code = contextWordVector.code.get(i);
							double prob = MathUtils.sigmoid(MathUtils.dotProduct(nodeVector.vector, inputWordVectorCopy));
							double gradient = (1 - code.getValue() - prob) * context.alpha;
							MathUtils.gradientUpdate(inputWordVector.vector, nodeVector.vector, gradient);
							MathUtils.gradientUpdate(nodeVector.vector, inputWordVectorCopy, gradient);
						}
					}
				}
			}
		},
		/** Continuous Bag of Words */
		CBOW {
			@Override
			public void trainOnSentence(String[] sentence, TrainingContext context) {
				for (int outputWordPos = 0; outputWordPos < sentence.length; outputWordPos++) {
					String outputWord = sentence[outputWordPos];
					if (isOovWord(outputWord)) continue; // skip OOV word
					
					WordVector outputWordVector = context.vocab.get(outputWord);
					int reducedWindow = ThreadLocalRandom.current().nextInt(context.window);
					int start = Math.max(0, outputWordPos - context.window + reducedWindow);
					int end = Math.min(sentence.length, outputWordPos + context.window + 1 - reducedWindow);
					
					double[] projection = new double[context.layerSize];
					
					for (int inputWordPos = start; inputWordPos < end; inputWordPos++) {
						if (inputWordPos == outputWordPos) continue;
						String inputWord = sentence[inputWordPos];
						if (isOovWord(inputWord)) continue;
						MathUtils.vecAdd(projection, context.vocab.get(inputWord).vector);
					}
					
					double[] updateVector = new double[context.layerSize];
					
					for (int i = 0; i < outputWordVector.getCodeLen(); i++) {
						NodeVector nodeVector = outputWordVector.points.get(i);
						Code code = outputWordVector.code.get(i);
						double prob = MathUtils.sigmoid(MathUtils.dotProduct(projection, nodeVector.vector));
						double gradient = (1 - code.getValue() - prob) * context.alpha;
						MathUtils.gradientUpdate(updateVector, nodeVector.vector, gradient);
						MathUtils.gradientUpdate(nodeVector.vector, projection, gradient);
					}
					
					for (int inputWordPos = start; inputWordPos < end; inputWordPos++) {
						if (inputWordPos == outputWordPos) continue;
						String inputWord = sentence[inputWordPos];
						if (isOovWord(inputWord)) continue;
						MathUtils.vecAdd(context.vocab.get(inputWord).vector, updateVector);
					}
				}
			}
			
		};		
		
		/** Train model on sentence */
		public abstract void trainOnSentence(String[] sentence, TrainingContext context);
	}

	/**
	 * {@link Word2VecWorker} represent workers that can be concurrently executed to train the word2vec model 
	 * using asynchronous stochastic gradient descent
	 */
	private class Word2VecWorker implements Runnable {
		private final TrainingContext context;
		private final Iterable<String> sentences;
		
		private Word2VecWorker(TrainingContext context, Iterable<String> sentences) {
			this.context = context;
			this.sentences = sentences;
		}
		
		@Override public void run() {
			trainChunk();
		}
		
		/** Train word2vec model on a chunk of data */
		private void trainChunk() {
			for (String sentence : sentences) {
				String[] words = tokenizer.tokenize(sentence);
				long crtWordCount = context.wordCount.addAndGet(replaceOovWords(words));
				if (sampling) {
					words = sampleWords(words);
				}

				long sentCount = context.sentenceCount.incrementAndGet();
				if (sentCount % 100 == 0) {
					context.alpha = Math.max(MIN_ALPHA, initAlpha * (1 - 1.0 * crtWordCount / context.totalWordCount));
					if (sentCount % DEBUG_THROTTLE_FACTOR == 0) {
						logger.info(String.format("%d sentences trained..", sentCount));
					}
				}
				model.trainOnSentence(words, context);
			}
		}
		
		/**
		 * Replace words that do not exist in the vocabulary (out of vocabulary or OOV words)
		 * with a constant string {@link #OOV_WORD} for performance
		 */
		private int replaceOovWords(String[] sentence) {
			int numValidWord = 0; 
			for (int i = 0; i < sentence.length; i++) {
				if (!context.vocab.containsKey(sentence[i])) {
					sentence[i] = OOV_WORD;
				} else {
					numValidWord++;
				}
			}
			return numValidWord;
		}
		
		/** @return Subset of the input containing the samples */
		private String[] sampleWords(String[] words) {
			List<String> sampled = Lists.newArrayList();
			for (String word : words) {
				if (isOovWord(word))
					continue;
				double samplingRate = context.vocab.get(word).getSamplingRate();
				if (samplingRate == 1 || ThreadLocalRandom.current().nextDouble() < samplingRate)
					sampled.add(word);
			}
			
			return sampled.toArray(new String[0]);
		}
	}
}
