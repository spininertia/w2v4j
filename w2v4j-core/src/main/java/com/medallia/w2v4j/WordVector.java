package com.medallia.w2v4j;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import com.google.common.collect.ImmutableList;
import com.medallia.w2v4j.utils.MathUtils;

/**
 * Vector representation for a word along with information about its path in the Huffman Tree
 */
public class WordVector implements Serializable {
	private static final long serialVersionUID = 0L;
	
	enum Code {
		LEFT(0),
		RIGHT(1);
		
		private final int value;
		
		private Code(final int newValue) { this.value = newValue; }
		
		public int getValue() { return value; }
	}
	private final double samplingRate;	// sampling probability to sample this word
	final double[] vector;
	final ImmutableList<NodeVector> points;
	final ImmutableList<Code> code;
	
	WordVector(int layerSize, double sampleThreshold, double frequency, List<NodeVector> points, List<Code> code) {
		this.samplingRate = computeSamplingRatio(sampleThreshold, frequency);
		this.vector = MathUtils.randomInitialize(layerSize);
		this.points = ImmutableList.copyOf(points);
		this.code = ImmutableList.copyOf(code);
	}
	
	private static double computeSamplingRatio(double sample, double frequency) {
		return Math.min(1, Math.sqrt(sample / frequency));
	}
	
	/** @return Rate at which this word should be sampled */
	double getSamplingRate() {
		return samplingRate;
	}
	
	/** Length of the Huffman code for this word */
	int getCodeLen() {
		return code.size();
	}
	
	@Override
	public boolean equals(Object other) {
		if (other == null) {
			return false;
		}
		
		if (!(other instanceof WordVector)) {
			return false;
		}
		
		WordVector otherWord = (WordVector) other;
		return samplingRate == otherWord.samplingRate 
				&& Arrays.equals(vector, otherWord.vector)
				&& points.equals(otherWord.points)
				&& code.equals(otherWord.code);
	}

	@Override
	public int hashCode() {
		int result = 17;
		long f = Double.doubleToLongBits(samplingRate);
		result = 31 * result + (int) (f ^ (f >>> 32));
		result = 31 * result + Arrays.hashCode(vector);
		result = 31 * result + points.hashCode();
		result = 31 * result + code.hashCode();
		return result;
	}
}
