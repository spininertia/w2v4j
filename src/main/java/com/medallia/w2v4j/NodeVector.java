package com.medallia.w2v4j;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Vector representation for inner node of Huffman Tree for Hierarchical Softmax.  
 */
public class NodeVector implements Serializable{
	private static final long serialVersionUID = 0L;
	
	double[] vector;
	
	public NodeVector(int layerSize) {
		vector = new double[layerSize]; 
	}
	
	@Override
	public boolean equals(Object other) {
		if (other == null) {
			return false;
		}
		
		if (!(other instanceof NodeVector)) {
			return false;
		}
		
		NodeVector node = (NodeVector) other;
		return Arrays.equals(vector, node.vector);
	}

	@Override
	public int hashCode() {
		int result = 17;
		result = 31 * result + Arrays.hashCode(vector);
		return result;
	}
}
