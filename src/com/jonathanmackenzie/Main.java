package com.jonathanmackenzie;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.linalg.lossfunctions.LossFunctions;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class Main {

    public static void main(String[] args) {
	// write your code here

        int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
        int miniBatchSize = 32;						//Size of mini batch to use when  training
        int exampleLength = 1000;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 75;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 1;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters /
                                                    // 50 tbptt length: 20 parameter updates per minibatch
        int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
        int nCharactersToSample = 300;				//Length of each sample to generate
        int numLabelClasses = 6;
        int nSteps = 1;
        CSVSequenceRecordReader featureReader = new CSVSequenceRecordReader();
        CSVSequenceRecordReader labelReader = new CSVSequenceRecordReader(nSteps);
        try {
            File infile = new File(args[0]);
            featureReader.initialize(new FileSplit(infile));
            labelReader.initialize(new FileSplit(infile));
        } catch (IOException e) {
            System.err.println("Could not find file "+e.getMessage());
            return;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        List<String> labels = new ArrayList<>(6);
        for (int i = 16; i <= 21; i++) {
            labels.add(Integer.toString(i));
        }
        labelReader.setLabels(labels);

        SequenceRecordReaderDataSetIterator iter = new SequenceRecordReaderDataSetIterator(
                featureReader,
                labelReader,
                miniBatchSize,
                numLabelClasses,
                true,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
            );

        int numInputs = labelReader.getLabels().size();
        int numOutputs = numInputs - 4; // we don't need the timestamp or derived as an output


        Schema inputDataSchema = new Schema.Builder()
                .addColumnString("timestamp")
                .addColumnInteger("16")
                .addColumnInteger("17")
                .addColumnInteger("18")
                .addColumnInteger("19")
                .addColumnInteger("20")
                .addColumnInteger("21")
                .build();
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .stringToTimeTransform("timestamp", "YYYY-MM-DD HH:mm:ss", DateTimeZone.UTC)
                .transform(new DeriveColumnsFromTimeTransform.Builder("timestamp")
                        .addIntegerDerivedColumn("hourOfDay", DateTimeFieldType.hourOfDay())
                        .build())
                .transform(new DeriveColumnsFromTimeTransform.Builder("timestamp")
                        .addIntegerDerivedColumn("minuteOfDay", DateTimeFieldType.minuteOfDay())
                        .build())
                .transform(new DeriveColumnsFromTimeTransform.Builder("timestamp")
                        .addIntegerDerivedColumn("dayOfWeek", DateTimeFieldType.dayOfWeek())
                        .build())
                .build();


        Schema outputSchema = tp.getFinalSchema();
        System.out.println(outputSchema);


        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Example");

        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> stringData = sc.textFile(args[0]);

        //We first need to parse this format. It's comma-delimited (CSV) format, so let's parse it using CSVRecordReader:
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(featureReader));

        //Now, let's execute the transforms we defined earlier:
        SparkTransformExecutor exec = new SparkTransformExecutor();
        JavaRDD<List<Writable>> processedData = exec.execute(parsedInputData, tp);

        //For the sake of this example, let's collect the data locally and print it:
        JavaRDD<String> processedAsString = processedData.map(new WritablesToStringFunction(","));

        List<String> processedCollected = processedAsString.collect();
        List<String> inputDataCollected = stringData.collect();


        System.out.println("\n\n---- Original Data ----");
        for(String s : inputDataCollected) System.out.println(s);

        System.out.println("\n\n---- Processed Data ----");
        for(String s : processedCollected) System.out.println(s);


//        tp.executeSequence(labelReader.sequenceRecord());
//        tp.executeSequence(featureReader.sequenceRecord());




        MultiLayerConfiguration netConf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .learningRate(0.01)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.RMSPROP)
                    .list()
                        .layer(0, new GravesLSTM.Builder().nIn(outputSchema.getColumnNames().size()).nOut(lstmLayerSize)
                                .activation("tanh").build())
                        .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                                .activation("tanh").build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
                                .nIn(lstmLayerSize).nOut(numOutputs).build())
                .backpropType(BackpropType.TruncatedBPTT)
                    .tBPTTForwardLength(tbpttLength)
                    .tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(netConf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        int nEpochs = 30;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
//        Logger log = LoggerFactory.getLogger("TrafficPredict");

        for (int i = 0; i < nEpochs; i++) {
            net.fit(iter);

            //Evaluate on the test set:
            Evaluation evaluation = net.evaluate(iter);
//            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));


            iter.reset();
        }

//        log.info("----- Finished -----");

    }
}
