package com.medallia.w2v4j.tokenizer;

/**
 * Interface for word tokenizer.
 */
public interface Tokenizer {
	public String[] tokenize(String sentence);
}
