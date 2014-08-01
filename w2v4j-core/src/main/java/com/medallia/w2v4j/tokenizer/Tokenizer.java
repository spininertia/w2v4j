package com.medallia.w2v4j.tokenizer;

/**
 * Interface for word tokenizer
 */
public interface Tokenizer {
	/** @return Words extracted from the given sentence */
	public String[] tokenize(String sentence);
}
