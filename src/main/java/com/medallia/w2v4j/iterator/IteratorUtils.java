package com.medallia.w2v4j.iterator;

import java.io.File;
import java.util.Iterator;

/**
 * Utilities for iterating through sentences. 
 */
public class IteratorUtils {
	
	public static Iterable<String> fileSentenceIterable(final File file) {
		return new Iterable<String>() {
			public Iterator<String> iterator() {
				return new LineSentenceIterator(file);
			}
		};
	}
}
