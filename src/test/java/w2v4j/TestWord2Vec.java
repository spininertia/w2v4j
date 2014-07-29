package w2v4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.junit.Test;

import com.google.common.base.Stopwatch;
import com.medallia.w2v4j.Word2VecModel;
import com.medallia.w2v4j.Word2VecTrainer;
import com.medallia.w2v4j.Word2VecTrainer.Word2VecTrainerBuilder;
import com.medallia.w2v4j.WordWithSimilarity;
import com.medallia.w2v4j.utils.SerializationUtils;

public class TestWord2Vec {
	
	static final String TRAIN_PATH = "src/main/resources/data/hilton_2014Q1_comments.segmented_cleaned";
	static final String MODEL_PATH = "src/main/resources/model/hilton.model";
	
	@Test
	public void testTrain() throws FileNotFoundException, IOException {
		Word2VecTrainer trainer = new Word2VecTrainerBuilder()
						.numWorker(Runtime.getRuntime().availableProcessors())
						.minCount(100)
						.build();
		Stopwatch stopwatch = Stopwatch.createStarted();
		Word2VecModel model = trainer.train(new File(TRAIN_PATH));
		stopwatch.stop();
		SerializationUtils.saveModel(model, MODEL_PATH);
		long elapsedTime = stopwatch.elapsed(TimeUnit.SECONDS);
		System.out.println("Time elapsed for trainnig(second):" + elapsedTime);
	}
	
	@Test
	public void testLoad() throws FileNotFoundException, ClassNotFoundException, IOException {
		Word2VecModel model = SerializationUtils.loadModel(MODEL_PATH);
		for (WordWithSimilarity word : model.mostSimilar("staff", 20)) {
			System.out.println(word);
		}
	}

}
