package com.medallia.w2v4j.tokenizer;

import java.io.Serializable;

public class RegexTokenizer implements Tokenizer, Serializable {
	
	public String[] tokenize(String sentence) {
		return sentence.split("\\s");
	}
	
}
