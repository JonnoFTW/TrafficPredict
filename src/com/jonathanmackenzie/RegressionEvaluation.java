package com.jonathanmackenzie;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by mack0242 on 30/08/16.
 */
public class RegressionEvaluation  {
    private int rowsSeen = 0;
    public RegressionEvaluation() {

    }

    /**
     *
     * @param actualData the true results
     * @param predictions our predictions about the results, should line up properly
     */
    public void eval(INDArray actualData, INDArray predictions) {

        rowsSeen += actualData.rows();
        // scale both data sets
        INDArray mean = actualData.mean(0);
        INDArray std  = actualData.std(0);
        INDArray scaledActual = actualData.subColumnVector(mean).divColumnVector(std);
        INDArray scaledPredictions = predictions.subColumnVector(mean).divColumnVector(std);
        // find the percentage difference
        INDArray diff = scaledActual.sub(scaledPredictions);
        // find the mean/median/  %error,

    }

    public String stats() {
        return String.format("Row Count: %d", rowsSeen);
    }
    public static void main(String[] args) {
        INDArray arrA = Nd4j.create(new double[][]{{1,2,3,4},{1,2,4,5},{2,3,4,4}}); // ACTUAL VALUES
        INDArray arrB= Nd4j.create(new double[][] {{2,3,3,5},{0,2,4,5},{3,3,3,4}}); // PREDICTED VALUES

        System.out.println("ACTUAL\n"+arrA);
        System.out.println("PREDICTED\n"+arrB);

        INDArray mean = arrA.mean(0), std = arrA.std(0);
        System.out.println("MEAN\n"+mean);
        System.out.println("STD\n"+std);

        INDArray scaledA = arrA.subColumnVector(mean).divColumnVector(std);
        System.out.println("SCALED\n"+scaledA);




    }
}

