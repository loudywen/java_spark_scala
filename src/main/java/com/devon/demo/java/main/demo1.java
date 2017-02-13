package com.devon.demo.java.main;
/**
 * Created by Devon on 2/12/2017.
 */

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.simple.Sentence;
import java.util.List;
import java.util.Properties;

public class demo1 {

  public static void main(String[] args) {
    // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
    props.setProperty("ner.model", "edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz");

    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

    // read some text in the text variable
    String text = "what time is it";

    // create an empty Annotation just with the given text
    Annotation document = new Annotation(text);

    // run all Annotators on this text
    pipeline.annotate(document);

  /*  List<CoreMap> sentences = document.get(SentencesAnnotation.class);

    for(CoreMap sentence: sentences) {
      // traversing the words in the current sentence
      // a CoreLabel is a CoreMap with additional token-specific methods
      for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
        // this is the text of the token
        String word = token.get(TextAnnotation.class);
//        System.out.println(word);
        // this is the POS tag of the token
        String pos = token.get(PartOfSpeechAnnotation.class);
//        System.out.println(pos);
        // this is the NER label of the token
        String ne = token.get(NamedEntityTagAnnotation.class);


      }

      // this is the parse tree of the current sentence
      Tree tree = sentence.get(TreeAnnotation.class);

      // this is the Stanford dependency graph of the current sentence
      SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);*/

      Sentence     sent    = new Sentence("where is the mexican restaurant in China");
      List<String> nerTags = sent.nerTags();
      System.out.println(nerTags);

    }


}
