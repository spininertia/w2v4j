package com.medallia.w2v4j.utils;

import com.google.common.base.Preconditions;

public class Utils {
	
	public static double dotProduct(double[] vec1, double[] vec2) {
		// TODO remove to improve performance
		Preconditions.checkArgument(vec1.length == vec2.length, "dimension not match");
		double product = 0;
		for (int i = 0; i < vec1.length; i++) {
			product += vec1[i] * vec2[i];
		}
		return product;
	}
	
	public static void gradientUpdate(double[] vec1, final double[] vec2, double gradient ) {
		for (int i = 0; i < vec1.length; i++) {
			vec1[i] += gradient * vec2[i];
		}
	}
	
	/** l2-norm */
	public static void normalize(double[] vec) {
		double vecLen = vecLen(vec);
		for (int i = 0; i < vec.length; i++) {
			vec[i] /= vecLen;
		}
	}
	
	public static double vecLen(double[] vec) {
		double l2 = 0;
		for (double val : vec) {
			l2 += val * val;
		}
		return Math.sqrt(l2);
	}
	
	public static double[] randomInitialize(int size) {
		double[] arr = new double[size];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = (Math.random() - 0.5) / size;
		}
		return arr;
	}
}
