package com.medallia.w2v4j;

import java.io.Serializable;
import java.util.List;

import com.google.common.collect.ImmutableList;
import com.medallia.w2v4j.utils.Utils;

public class WordVector implements Serializable{
	enum Code {
		LEFT(0),
		RIGHT(1);
		
		private final int value;
		
		private Code(final int newValue) { this.value = newValue; }
		
		public int getValue() { return value; }
	}
	
	double[] vector;
	ImmutableList<NodeVector> points;
	ImmutableList<Code> code;
	
	public WordVector(int layerSize, List<NodeVector> points, List<Code> code) {
		vector = Utils.randomInitialize(layerSize);
		this.points = ImmutableList.copyOf(points);
		this.code = ImmutableList.copyOf(code);
	}
	
	public int getCodeLen() {
		return code.size();
	}
	
}
