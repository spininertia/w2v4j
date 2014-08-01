package com.medallia.w2v4j;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.base.Predicates;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Ordering;
import com.medallia.w2v4j.utils.MathUtils;

/**
 * {@code Word2VecModel} defines a trained Word2Vec model
 */
public class Word2VecModel implements Serializable {
	private static final long serialVersionUID = 0L;
	
	private final ImmutableMap<String, WordVector> vocab;	// map from word to its WordNeuron
	
	Word2VecModel(ImmutableMap<String, WordVector> vocab) {
		this.vocab = vocab;
	}
	
	/** @return Cosine similarity between the given two words */
	public double similarity(String word1, String word2) {
		if (word1 == null || word2 == null)
			return 0;
		
		if (word1.equals(word2))
			return 1;
		
		WordVector neuron1 = vocab.get(word1);
		WordVector neuron2 = vocab.get(word2);
		
		if (neuron1 == null || neuron2 == null)
			return 0;
		
		// this computes cosine similarity since vectors are already normalized
		return MathUtils.dotProduct(neuron1.vector, neuron2.vector);
	}
	
	/** @return At most n similar words to the given word along with their similarity */
	public List<WordWithSimilarity> mostSimilar(final String word, int n) {
		if (!vocab.containsKey(word))
			return ImmutableList.of();
		
		return Ordering.natural().greatestOf(
				FluentIterable.from(vocab.keySet())
					.filter(Predicates.not(Predicates.equalTo(word)))
					.transform(new Function<String, WordWithSimilarity>() {
						@Override public WordWithSimilarity apply(String other) {
							return new WordWithSimilarity(other, similarity(word, other));
						}
					}),
				n
			);
	}
	
	/** @return whether word is in vocabulary */
	public boolean containsWord(String word) {
		return vocab.containsKey(word);
	}
	
	/** @return defensive copy of the word vector */
	public double[] getWordVector(String word) {
		if (!containsWord(word))
			return new double[0];
		double[] vector = vocab.get(word).vector;
		return Arrays.copyOf(vector, vector.length);
	}
	
	@Override
	public boolean equals(Object other) {
		if (other == null) {
			return false;
		}
		
		if (!(other instanceof Word2VecModel)) {
			return false;
		}
		
		return this.vocab.equals(((Word2VecModel)other).vocab);
		
	}

	@Override
	public int hashCode() {
		int result = 17;
		result = 31 * result + vocab.hashCode();
		return result;
	}
}
