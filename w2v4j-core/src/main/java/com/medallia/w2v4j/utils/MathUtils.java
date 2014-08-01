package com.medallia.w2v4j.utils;

import com.google.common.base.Preconditions;

/**
 * Utilities for math computation.
 */
public class MathUtils {
	
	/** 
	 * @return dot product of two given vectors 
	 */
	public static double dotProduct(double[] vec1, double[] vec2) {
		Preconditions.checkArgument(vec1.length == vec2.length, "dimension not match");
		double product = 0;
		for (int i = 0; i < vec1.length; i++) {
			product += vec1[i] * vec2[i];
		}
		return product;
	}
	
	/**
	 * Add two vectors, result is saved in the first vector
	 */
	public static void vecAdd(double[] vec1, double[] vec2) {
		for (int i = 0; i < vec1.length; i++) {
			vec1[i] += vec2[i];
		}
	}
	
	/** 
	 * Scalar divide of the given vector
	 * Note that this mutates the input for performance 
	 */
	public static void vecDivide(double vec[], double num) {
		for (int i = 0; i < vec.length; i++) {
			vec[i] /= num;
		}
	}
	
	/** Gradient update */
	public static void gradientUpdate(double[] vecToUpdate, final double[] vec, double gradient) {
		for (int i = 0; i < vecToUpdate.length; i++) {
			vecToUpdate[i] += gradient * vec[i];
		}
	}
	
	/** 
	 * L2-normalize a vector
	 * Note that this mutates the input for performance
	 */
	public static void normalize(double[] vec) {
		double vecLen = vecLen(vec);
		for (int i = 0; i < vec.length; i++) {
			vec[i] /= vecLen;
		}
	}
	
	/** 
	 * Compute L2-norm of vector. 
	 */
	public static double vecLen(double[] vec) {
		double l2 = 0;
		for (double val : vec) {
			l2 += val * val;
		}
		return Math.sqrt(l2);
	}
	
	/** @return a double vector whose values are randomly chosen from (0, 1) */
	public static double[] randomInitialize(int size) {
		double[] arr = new double[size];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = (Math.random() - 0.5) / size;
		}
		return arr;
	}
	
	/** Sigmoid activation function */
	public static double sigmoid(double x) {
		return 1.0 / (1 + Math.exp(-x));
	}
}
