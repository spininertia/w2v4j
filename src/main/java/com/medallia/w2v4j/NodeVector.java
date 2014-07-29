package com.medallia.w2v4j;

import java.io.Serializable;

/**
 * Vector representation for inner node of Huffman Tree for Hierarchical Softmax.  
 */
public class NodeVector implements Serializable{
	private static final long serialVersionUID = 0L;
	
	double[] vector;
	
	public NodeVector(int layerSize) {
		vector = new double[layerSize]; 
	}
}
