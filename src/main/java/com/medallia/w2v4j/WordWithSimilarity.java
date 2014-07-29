package com.medallia.w2v4j;

/**
 * Word with its similarity. 
 */
public class WordWithSimilarity implements Comparable<WordWithSimilarity>{
	private final String word;
	private final double similarity;
	
	public WordWithSimilarity(String word, double similarity) {
		this.word = word;
		this.similarity = similarity;
	}
	
	public String getWord() {
		return word;
	}
	
	public double getSimilarity() {
		return similarity;
	}
	

	public int compareTo(WordWithSimilarity word) {
		if (this.similarity < word.similarity) {
			return -1;
		}
		return 1;
	}
	
	@Override
	public String toString() {
		return word + "\t" + similarity;
	}
}
