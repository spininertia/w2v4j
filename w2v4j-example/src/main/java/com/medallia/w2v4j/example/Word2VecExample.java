package com.medallia.w2v4j.example;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import com.medallia.w2v4j.Word2VecModel;
import com.medallia.w2v4j.Word2VecTrainer;
import com.medallia.w2v4j.Word2VecTrainer.NeuralNetworkLanguageModel;
import com.medallia.w2v4j.Word2VecTrainer.Word2VecTrainerBuilder;
import com.medallia.w2v4j.WordWithSimilarity;
import com.medallia.w2v4j.utils.SerializationUtils;

public class Word2VecExample {
	public static final String TRAIN_PATH = "";
	public static final String MODEL_PATH = "";
	
	public static void train() throws FileNotFoundException, IOException {
		Word2VecTrainer trainer = new Word2VecTrainerBuilder()
									.minCount(0)
									.numWorker(Runtime.getRuntime().availableProcessors())
									.model(NeuralNetworkLanguageModel.SKIP_GRAM)
									.build();
		Word2VecModel model = trainer.train(new File(TRAIN_PATH));
		SerializationUtils.saveModel(model, MODEL_PATH);
	}
	
	public static void test() throws FileNotFoundException, ClassNotFoundException, IOException {
		Word2VecModel model = SerializationUtils.loadModel(MODEL_PATH);
		for (WordWithSimilarity word : model.mostSimilar("sentence", 10)) {
			System.out.println(word.getWord() + "\t" + word.getSimilarity());
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
		train();
		test();
	}
}
