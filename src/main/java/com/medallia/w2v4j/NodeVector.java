package com.medallia.w2v4j;

import java.io.Serializable;

public class NodeVector implements Serializable{
	double[] vector;
	public NodeVector(int layerSize) {
		vector = new double[layerSize]; 
	}
}
