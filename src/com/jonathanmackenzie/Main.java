package com.jonathanmackenzie;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.reflections.Reflections.collect;


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
        double splitPortion = 0.66 ;// 66% training data, 33% testing data
//        CSVSequenceRecordReader featureReader = new CSVSequenceRecordReader();
//        CSVSequenceRecordReader labelReader = new CSVSequenceRecordReader(nSteps);
//        try {
//            File infile = new File(args[0]);
//            featureReader.initialize(new FileSplit(infile));
//            labelReader.initialize(new FileSplit(infile));
//        } catch (IOException e) {
//            System.err.println("Could not find file "+e.getMessage());
//            return;
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }

//        SequenceRecordReaderDataSetIterator iter = new SequenceRecordReaderDataSetIterator(
//                featureReader,
//                labelReader,
//                miniBatchSize,
//                numLabelClasses,
//                true,
//                SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH
//            );



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
                .
                .build();

        Schema outputSchema = tp.getFinalSchema();
        System.out.println(outputSchema);


        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");

        conf.setAppName("Traffic Predict");

        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");
        JavaRDD<String> stringData = sc.textFile(new File(args[0]).getAbsolutePath());
        RecordReader rr = new CSVRecordReader();

        //We first need to parse this format. It's comma-delimited (CSV) format, so let's parse it using CSVRecordReader:
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

        //Now, let's execute the transforms we defined earlier:
        SparkTransformExecutor exec = new SparkTransformExecutor();
        JavaRDD<List<Writable>> processedData = exec.execute(parsedInputData, tp);
        processedData.cache();
        long finalDataCount = processedData.count();

        int numInputs = tp.getFinalSchema().getColumnNames().size();
        int numOutputs = numInputs - 4; // we don't need the timestamp or derived as an output

        System.out.println("DATA COUNT: "+finalDataCount);
        List<List<Writable>> datalist = processedData.collect();
        INDArray input  = Nd4j.zeros(numInputs, (int)finalDataCount);
        INDArray labels = Nd4j.zeros(numInputs, (int)finalDataCount);
        int rowCount = 0;
        for(List<Writable> row : datalist) {
           // input.putScalar(new int[]{row.to)});
          //  labels.putScalar();
        }
        sc.stop();

        DataSet ds = new DataSet(input, labels);
        SplitTestAndTrain splitTestAndTrain = ds.splitTestAndTrain(splitPortion);
        DataSet train = splitTestAndTrain.getTrain();
        DataSet test  = splitTestAndTrain.getTest();


        MultiLayerConfiguration netConf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .learningRate(0.01)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.RMSPROP)
                    .list()
                        .layer(0, new GravesLSTM.Builder().nIn(outputSchema.getColumnNames().size()).nOut(lstmLayerSize).name("input")
                                .activation("tanh").build())
                        .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).name("hidden")
                                .activation("tanh").build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax").name("output")        //MCXENT + softmax for classification
                                .nIn(lstmLayerSize).nOut(numOutputs).build())
                .backpropType(BackpropType.TruncatedBPTT)
                    .tBPTTForwardLength(tbpttLength)
                    .tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(netConf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));


        net.fit(train);
        net.evaluate(test.iterateWithMiniBatches());
//        System.out.println("Executing tp");
////         tp.execute(labelReader.next());
//        SplitTestAndTrain testAndTrain = ds.splitTestAndTrain(0.65);
//        DataSet testing = testAndTrain.getTest();
//        DataSet training = testAndTrain.getTrain();



//        int nEpochs = 30;
//        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
//        Logger log = LoggerFactory.getLogger("TrafficPredict");
//
//        net.fit(training);
        //Evaluation eval = Evaluation.evalTimeSeries();
//        for (int i = 0; i < nEpochs; i++) {
//            net.fit(testing);

            //Evaluate on the test set:
//            Evaluation evaluation = net.evaluate(iter);
//            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));
//
//
//            iter.reset();
//        }

//        log.info("----- Finished -----");

    }
}
