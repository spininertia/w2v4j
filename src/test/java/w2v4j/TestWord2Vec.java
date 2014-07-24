package w2v4j;

import java.io.File;
import java.util.concurrent.TimeUnit;

import org.junit.Test;

import com.google.common.base.Stopwatch;
import com.medallia.w2v4j.Word2Vec;
import com.medallia.w2v4j.WordWithSimilarity;

public class TestWord2Vec {

	@Test
	public void testTrain() {
		Word2Vec model = new Word2Vec(new File("src/main/resources/data/hilton_2014Q1_comments.segmented_cleaned"));
		Stopwatch stopwatch = Stopwatch.createStarted();
		model.train();
		model.save("src/main/resources/model/hilton.model");
		stopwatch.stop();
		long elapsedTime = stopwatch.elapsed(TimeUnit.SECONDS);
		System.out.println(elapsedTime);
	}
	
	@Test
	public void testLoad() {
		Word2Vec model = Word2Vec.load("src/main/resources/model/hilton.model");
		for (WordWithSimilarity word : model.mostSimilar("staff", 50)) {
			System.out.println(word);
		}
	}

}
