package w2v4j;

import java.util.concurrent.TimeUnit;

import org.junit.Test;

import com.google.common.base.Stopwatch;
import com.medallia.w2v4j.Word2Vec;
import com.medallia.w2v4j.Word2Vec.Word2VecBuilder;
import com.medallia.w2v4j.WordWithSimilarity;

public class TestWord2Vec {
	
	static final String TRAIN_PATH = "src/main/resources/data/hilton_2014Q1_comments.segmented_cleaned";
	static final String MODEL_PATH = "src/main/resources/model/hilton.model";
	@Test
	public void testTrain() {
		Word2Vec model = new Word2VecBuilder(TRAIN_PATH)
						.minCount(100)
						.build();
		Stopwatch stopwatch = Stopwatch.createStarted();
		model.train();
		model.save(MODEL_PATH);
		stopwatch.stop();
		long elapsedTime = stopwatch.elapsed(TimeUnit.SECONDS);
		System.out.println(elapsedTime);
	}
	
	@Test
	public void testLoad() {
		Word2Vec model = Word2Vec.load(MODEL_PATH);
		for (WordWithSimilarity word : model.mostSimilar("staff", 20)) {
			System.out.println(word);
		}
	}

}
