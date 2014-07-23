package com.medallia.w2v4j;

import java.io.Serializable;

public class NodeNeuron implements Serializable{
	double[] vector;
	public NodeNeuron(int layerSize) {
		vector = new double[layerSize]; // TODO: may need random init here
		
	}
}
