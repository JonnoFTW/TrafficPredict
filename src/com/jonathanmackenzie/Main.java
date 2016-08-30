package com.jonathanmackenzie;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    public static void main(String[] args) {
	// write your code here

        int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
        int miniBatchSize = 32;						//Size of mini batch to use when  training
        int exampleLength = 1000;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 75;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 10;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters /
                                                    // 50 tbptt length: 20 parameter updates per minibatch
        int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
        int nCharactersToSample = 300;				//Length of each sample to generate
        int numLabelClasses = 6;
        int nSteps = 2;
        int labelIdx = 4;
        double splitPortion = 0.66 ;// 66% training data, 33% testing data


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
        int numOutputs = numInputs - labelIdx; // we don't need the timestamp or derived as an output

        List<List<Writable>> datalist = processedData.collect();
        INDArray input  = Nd4j.zeros((int)finalDataCount - nSteps, numInputs, 'c');
        INDArray labels = Nd4j.zeros((int)finalDataCount - nSteps, numOutputs, 'c');
        int rowCount = 0;
        for(List<Writable> row : datalist) {
            double[] inputData = new double[row.size()];
            double[] labelData = new double[numOutputs];
            List<Writable> labelRow;
            try {
                labelRow = datalist.get(rowCount + nSteps);
            } catch (IndexOutOfBoundsException e) {
                break;
            }
            for (int i = 0; i < inputData.length; i++) {
                inputData[i] = row.get(i).toDouble();
            }
            for (int i = 0; i < labelData.length; i++) {
                labelData[i] = labelRow.get(i+labelIdx).toDouble();
            }

            input.putRow(rowCount, Nd4j.create(inputData));
            labels.putRow(rowCount, Nd4j.create(labelData));
            rowCount++;
        }
        sc.stop();

        DataSet ds = new DataSet(input, labels);
        ds.setLabelNames(Arrays.asList("16,17,18,19,20,21".split(",")));
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
                        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation("identity").name("output")
                                .nIn(lstmLayerSize).nOut(numOutputs).build())
                .backpropType(BackpropType.TruncatedBPTT)
                    .tBPTTForwardLength(tbpttLength)
                    .tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(netConf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        Evaluation eval = new Evaluation();
        for(int i =0; i < numEpochs; i++) {
            System.out.println("------ TRAINING -----");
            net.fit(train);
            System.out.println("------ EVALUATING -----");

            INDArray out = net.output(test.getFeatures());
            int[] shape = test.getLabels().shape();

            eval.evalTimeSeries(test.getLabels().reshape(shape[0], shape[1], 1), out);
//        System.out.println("Predictions: "+out);
//        System.out.println("Expected:    "+test.getLabels());

            System.out.println(eval.stats());

        }

//;        INDArray output = net.output(test.getFeatureMatrix());
//        eval.evalTimeSeries(test.getFeatures(), output );
//        System.out.println(eval.stats());
//        test.get(10);
//
//        int nEpochs = 30;
//        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
//        Logger log = LoggerFactory.getLogger("TrafficPredict");
//
//        for (int i = 0; i < nEpochs; i++) {
//            net.fit(train);
//
//            // Evaluate on the test set:
//            Evaluation evaluation = net.evaluate(test);
//            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));
//
//        }
//
//        log.info("----- Finished -----");

    }
}
