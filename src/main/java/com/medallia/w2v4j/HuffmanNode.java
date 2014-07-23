package com.medallia.w2v4j;

import java.io.Serializable;

import com.sun.istack.internal.Nullable;

public class HuffmanNode implements Comparable<HuffmanNode>, Serializable{
	String word; // nullable, inner node don't have such field. 
	long count;
	HuffmanNode left;
	HuffmanNode right;
	
	public HuffmanNode(@Nullable String word, long count) {
		this.word = word;
		this.count = count;
		left = right = null;
	}

	public int compareTo(HuffmanNode node) {
		long diff = this.count - node.count;
		
		if (diff > 0) {
			return 1;
		} else if (diff < 0){ 
			return -1;
		}
		return 0;
	}
	
	@Override
	public String toString() {
		return String.format("%s %d", word == null ? "inner" : word, count);
	}
}
