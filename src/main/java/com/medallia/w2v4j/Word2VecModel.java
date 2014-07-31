package com.medallia.w2v4j;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.medallia.w2v4j.utils.MathUtils;

/**
 * {@code Word2VecModel} defines a trained Word2Vec model. 
 * It includes model itself, hyper-parameters along with the method to access the model. 
 */
public class Word2VecModel implements Serializable{
	private static final long serialVersionUID = 0L;
	
	final int window;			// window size 
	final int layerSize;		// dimension of word vector
	final int minCount; 		// the minimum frequency that a word in vocabulary needs to satisfy
	final double initAlpha;		// initial learning rate
	final boolean sg;			// apply skip-gram model if true, cbow other wise
	final boolean sampling;		// enable sub-sampling
	final double samplingThreshold; 
	
	long wordCount; // number of total words trained on
	ImmutableMap<String, WordVector> vocab;	// map from word to its WordNeuron
	
	Word2VecModel(Word2VecTrainer trainer) {
		this.window = trainer.window;
		this.layerSize = trainer.layerSize;
		this.minCount = trainer.minCount;
		this.initAlpha = trainer.initAlpha;
		this.sg  = trainer.sg;
		this.sampling = trainer.sampling;
		this.samplingThreshold = trainer.samplingThreshold;
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
		return MathUtils.dotProduct(neuron1.vector, neuron2.vector);
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
		return Ordering.natural().greatestOf(result, n);
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
}
