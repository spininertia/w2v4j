package w2v4j;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.junit.Test;

import com.google.common.collect.ImmutableList;
import com.medallia.w2v4j.Word2VecModel;
import com.medallia.w2v4j.Word2VecTrainer;
import com.medallia.w2v4j.Word2VecTrainer.Word2VecTrainerBuilder;

public class TestWord2Vec {
	
	/**
	 * Assert serialization works correctly. 
	 */
	@Test
	public void testSerialization() throws IOException, ClassNotFoundException {
		ImmutableList<String> sentences = ImmutableList.of("machine learning is awesome", "word2vec is fun");
		Word2VecTrainer trainer = new Word2VecTrainerBuilder()
			.minCount(0)
			.build();
		Word2VecModel model = trainer.train(sentences);
		
		ByteArrayOutputStream byteArray = new ByteArrayOutputStream();
		ObjectOutputStream out = new ObjectOutputStream(byteArray);
		out.writeObject(model);
		
		ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(byteArray.toByteArray()));
		Word2VecModel deserialized = (Word2VecModel) in.readObject();
		assertEquals(deserialized, model);
	}
}
