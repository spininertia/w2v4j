package com.medallia.w2v4j;

/**
 * Wrapper for a word and its similarity to some other word 
 */
public class WordWithSimilarity implements Comparable<WordWithSimilarity>{
	private final String word;
	private final double similarity;
	
	WordWithSimilarity(String word, double similarity) {
		this.word = word;
		this.similarity = similarity;
	}
	
	public String getWord() {
		return word;
	}
	
	public double getSimilarity() {
		return similarity;
	}
	
	@Override
	public int compareTo(WordWithSimilarity word) {
		return this.similarity < word.similarity ? -1 : 1;
	}
	
	@Override
	public String toString() {
		return String.format("%s\t%s", word, similarity);
	}
}
