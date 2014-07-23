package com.medallia.w2v4j.iterator;

import java.io.File;
import java.util.Iterator;

public interface SentenceIteratorFactory {
	Iterator<String> createSentenceIterator(File file);
}
