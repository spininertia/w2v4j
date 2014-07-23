package com.medallia.w2v4j.iterator;

import java.io.File;
import java.io.Serializable;
import java.util.Iterator;

public class LineSentenceIteratorFactory implements SentenceIteratorFactory, Serializable {
	
	static private LineSentenceIteratorFactory factory = new LineSentenceIteratorFactory();
	
	private LineSentenceIteratorFactory() {}
	
	public static SentenceIteratorFactory getInstance() {
		return factory;
	}
	
	public Iterator<String> createSentenceIterator(File file)  {
		return new LineSentenceIterator(file);
	}


}
