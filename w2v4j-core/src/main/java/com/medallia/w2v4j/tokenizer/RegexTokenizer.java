package com.medallia.w2v4j.tokenizer;

import java.util.regex.Pattern;


/** Tokenizer using simple regular expression. */
public class RegexTokenizer implements Tokenizer {
	
	private static final Pattern p = Pattern.compile("\\s");
	
	@Override
	public String[] tokenize(String sentence) {
		return p.split(sentence);
	}
	
}
