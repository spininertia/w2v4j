package com.medallia.w2v4j;

import java.io.Serializable;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import com.google.common.collect.ImmutableList;
import com.medallia.w2v4j.utils.MathUtils;

/**
 * Vector representation for a word along with information about its path in the Huffman Tree.
 */
public class WordVector implements Serializable{
	private static final long serialVersionUID = 0L;
	
	enum Code {
		LEFT(0),
		RIGHT(1);
		
		private final int value;
		
		private Code(final int newValue) { this.value = newValue; }
		
		public int getValue() { return value; }
	}
	
	double samplingRate;	// sampling probability to sample this word
	double[] vector;
	ImmutableList<NodeVector> points;
	ImmutableList<Code> code;
	
	public WordVector(int layerSize, double sampleThreshold, double frequency, List<NodeVector> points, List<Code> code) {
		this.samplingRate = computeSamplingRatio(sampleThreshold, frequency);
		this.vector = MathUtils.randomInitialize(layerSize);
		this.points = ImmutableList.copyOf(points);
		this.code = ImmutableList.copyOf(code);
	}
	
	private static double computeSamplingRatio(double sample, double frequency) {
		return Math.min(1, Math.sqrt(sample / frequency));
	}
	
	/** Returns true if the word is selected while sampling. */
	boolean sample() {
		return samplingRate == 1 ? true : ThreadLocalRandom.current().nextDouble() < samplingRate;
	}
	
	int getCodeLen() {
		return code.size();
	}
	
}
