package com.medallia.w2v4j.tokenizer;

import java.io.Serializable;

/** Tokenizer using simple regular expression. */
public class RegexTokenizer implements Tokenizer, Serializable {
	
	public String[] tokenize(String sentence) {
		return sentence.split("\\s");
	}
	
}
