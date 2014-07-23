package com.medallia.w2v4j;

import java.io.Serializable;

public class WordWithSimilarity implements Serializable, Comparable<WordWithSimilarity>{
	String word;
	double similarity;
	
	public WordWithSimilarity(String word, double similarity) {
		this.word = word;
		this.similarity = similarity;
	}

	public int compareTo(WordWithSimilarity word) {
		if (this.similarity < word.similarity) {
			return 1;
		}
		return -1;
	}
	
	@Override
	public String toString() {
		return word + "\t" + similarity;
	}
}
