package com.medallia.w2v4j.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;

/** Split file into chunks. */
public class FileSplitter {
	static Logger logger = LogManager.getRootLogger();
	static final String CHUNK_PATH = "src/main/resources/tmp";
	
	static {
		File dir = new File(CHUNK_PATH); 
		if (!dir.exists()) {
			dir.mkdir();
		}
	}
	
	public static ImmutableList<String> chunkByLine(String filePath, int numChunk) throws IOException {
		if (numChunk <= 1) {
			return ImmutableList.of(filePath);
		}
		List<String> chunks = Lists.newArrayList();
		
		long numLine = countLines(filePath);
		long linePerChunk = numLine / numChunk;
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		
		long start = 0;
		for (int i = 1; i <= numChunk; i++) {
			String chunkName = new File(CHUNK_PATH, new File(filePath).getName() + "_" + i + ".chunk").toString();
			PrintWriter writer = new PrintWriter(chunkName);
			chunks.add(chunkName);
			
			long end  = i == numChunk ? numLine : start + linePerChunk;
			long j = start;
			while (j < end) {
				j++;
				writer.println(reader.readLine());
			}
			writer.close();
			start += linePerChunk; 
		}
		reader.close();
		
		logger.info(String.format("Chunk completed. Number of lines per chunk: %d", linePerChunk));
		return ImmutableList.copyOf(chunks);
	}
	
	
	private static long countLines(String file) throws IOException {
		long lines = 0;
		BufferedReader reader = new BufferedReader(new FileReader(file));
		while (reader.readLine() != null) lines++;
		reader.close();
		return lines;
	}
}
