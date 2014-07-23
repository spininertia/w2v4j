package com.medallia.w2v4j.iterator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * @{link DeafultSentenceIterator} iterates through file line by line, assuming each line is a sentence.
 */
public class LineSentenceIterator implements Iterator<String>{
	
	private BufferedReader reader;
	private String cachedLine = null;
	private boolean finished = false;
	
	public LineSentenceIterator(File inputFile)  {
		try {
			reader = new BufferedReader(new FileReader(inputFile));
		} catch (FileNotFoundException e) {
			System.err.println("File not found!");
			System.exit(1);
		}
	}
	
	public boolean hasNext() {
		if (cachedLine != null) {
			return true;
		}
		
		if (finished) {
			return false;
		}
		
		try {
			cachedLine = reader.readLine();
			if (cachedLine == null) {
				safeClose();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return cachedLine != null;
	}

	public String next() {
		if (hasNext()) {
			String line = cachedLine;
			cachedLine = null;
			return line;
		}
		throw new NoSuchElementException();
	}

	public void remove() {
		throw new UnsupportedOperationException();
	}
	
	public void safeClose() {
		try {
			finished = true; 
			reader.close();
		} catch (IOException e) {
			
		}
	}

}
