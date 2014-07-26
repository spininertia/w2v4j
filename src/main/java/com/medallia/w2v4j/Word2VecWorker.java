package com.medallia.w2v4j;

import java.io.File;
import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.medallia.w2v4j.WordNeuron.Code;
import com.medallia.w2v4j.utils.Utils;

public class Word2VecWorker implements Runnable {
	static Logger logger = LogManager.getRootLogger();
	
	int workerId;
	File corpusChunk;
	Word2Vec model;
	
	public Word2VecWorker(int workerId, Word2Vec model, String chunk) {
		this.workerId = workerId;
		this.model = model;
		this.corpusChunk = new File(chunk);
	}
	
	public void run() {
		trainChunk();
	}
	
	private void trainChunk() {
		Iterator<String> sentenceIterator = model.sentenceIteratorFactory.createSentenceIterator(corpusChunk);
		while (sentenceIterator.hasNext()) {
			String[] words = model.tokenizer.tokenize(sentenceIterator.next());
			long crtWordCount = model.wordCount.addAndGet(model.replaceOovWords(words));
			long sentCount = model.sentenceCount.incrementAndGet();
			if (sentCount % 100 == 0) {
				model.alpha = Math.max(Word2Vec.MIN_ALPHA, model.initAlpha * (1 - 1.0 * crtWordCount / model.totalWords));
				if (sentCount % 10000 == 0) {
					logger.info(String.format("%d sentences trained..", sentCount));
				}
			}
			trainSkipGramForSentence(model.alpha, words);
		}
	}
	
	private void trainSkipGramForSentence(double alpha, String[] sentence) {
		for (int contextWordPos = 0; contextWordPos < sentence.length; contextWordPos++) {
			String contextWord = sentence[contextWordPos];
			// skip OOV word
			if (model.isOovWord(contextWord)) continue;
			
			WordNeuron contextWordNeuron = model.vocab.get(contextWord);
			int reducedWindow = ThreadLocalRandom.current().nextInt(model.window);
			int start = Math.max(0, contextWordPos - model.window + reducedWindow);
			int end = Math.min(sentence.length, contextWordPos + model.window + 1 - reducedWindow);
			
			for (int inputWordPos = start; inputWordPos < end; inputWordPos++) {
				if (inputWordPos == contextWordPos) continue;
				
				String inputWord = sentence[inputWordPos];
				// skip OOV word
				if (model.isOovWord(inputWord)) continue;
				WordNeuron inputWordNeuron = model.vocab.get(inputWord);
				
				double[] inputWordVector = Arrays.copyOf(inputWordNeuron.vector, model.layerSize);
				for (int i = 0; i < contextWordNeuron.getCodeLen(); i++) {
					NodeNeuron nodeNeuron = contextWordNeuron.points.get(i);
					Code code = contextWordNeuron.code.get(i);
					double prob = 1.0 / (1 + Math.exp(-Utils.dotProduct(nodeNeuron.vector, inputWordVector))); // TODO change exp to table lookup to speed-up
					double gradient = (1 - code.getValue() - prob) * alpha;
					// TODO atomic long array
					Utils.gradientUpdate(inputWordNeuron.vector, nodeNeuron.vector, gradient);
					Utils.gradientUpdate(nodeNeuron.vector, inputWordVector, gradient);
				}
			}
		}
	}


}
