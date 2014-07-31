package com.medallia.w2v4j.tokenizer;


/** Tokenizer using simple regular expression. */
public class RegexTokenizer implements Tokenizer{
	
	@Override
	public String[] tokenize(String sentence) {
		return sentence.split("\\s");
	}
	
}
