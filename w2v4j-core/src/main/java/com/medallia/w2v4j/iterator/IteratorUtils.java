package com.medallia.w2v4j.iterator;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import org.apache.commons.io.FileUtils;

/**
 * Utilities for iterating through sentences. 
 */
public class IteratorUtils {
	/** @return {@link Iterable} for the newline delimited file */
	public static Iterable<String> fileSentenceIterable(final File file) {
		return new Iterable<String>() {
			@Override
			public Iterator<String> iterator() {
				try {
					return FileUtils.lineIterator(file);
				} catch (IOException e) {
					throw new IllegalStateException("Failed to read file " + file, e);
				}
			}
		};
	}
}
