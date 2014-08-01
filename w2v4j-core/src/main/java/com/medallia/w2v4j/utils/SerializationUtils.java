package com.medallia.w2v4j.utils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import com.medallia.w2v4j.Word2VecModel;

/**
 * Utility class for serialization and deserialization of {@code Word2VecModel}. 
 */
public class SerializationUtils {
	
	/** Serialize and save the {@link Word2VecModel} onto disk. */
	public static void saveModel(Word2VecModel model, String path) throws FileNotFoundException, IOException { 
		ObjectOutputStream out;
		out = new ObjectOutputStream(new FileOutputStream(path));
		out.writeObject(model);
		out.close();
	}
	
	/** Load and deserialize {@link Word2VecModel}. */
	public static Word2VecModel loadModel(String path) throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream input = new ObjectInputStream(new FileInputStream(path));
		Word2VecModel model = (Word2VecModel) input.readObject();
		input.close();
		return model;
	}
}
