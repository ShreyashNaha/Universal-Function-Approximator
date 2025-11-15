package ifndef.nn.core;

import ifndef.nn.activation.ActivationFunction;
import java.util.Random;

public class DenseLayer implements Layer {

    private final int inputSize;
    private final int outputSize;
    private final ActivationFunction activation;

    private double[][] weights;
    private double[] biases;

    private transient double[] lastInputs;
    private transient double[] lastWeightedSums;

    public DenseLayer(int inputSize, int outputSize, ActivationFunction activation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;

        this.weights = new double[outputSize][inputSize];
        this.biases = new double[outputSize];

        Random rand = new Random();
        double variance = Math.sqrt(2.0 / (inputSize + outputSize));
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                this.weights[i][j] = rand.nextGaussian() * variance;
            }
            this.biases[i] = 0.0;
        }
    }

    @Override
    public double[] forward(double[] inputs) {
        
        this.lastInputs = inputs;

        double[] outputs = new double[outputSize];
        this.lastWeightedSums = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            double weightedSum = 0.0;
            for (int j = 0; j < inputSize; j++) {
                weightedSum += inputs[j] * weights[i][j];
            }
            weightedSum += biases[i];
            
            this.lastWeightedSums[i] = weightedSum;

            if (activation != null) {
                outputs[i] = activation.value(weightedSum);
            } else {
                outputs[i] = weightedSum;
            }
        }
        return outputs;
    }

    @Override
    public GradientPackage backward(double[] outputGradient) {
        double[] layerGradient = new double[outputSize];
        double[] inputGradient = new double[inputSize];
        double[][] weightGradients = new double[outputSize][inputSize];
        double[] biasGradients = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            if (activation != null) {
                layerGradient[i] = outputGradient[i] * activation.derivative(lastWeightedSums[i]);
            } else {
                layerGradient[i] = outputGradient[i];
            }
        }

        for (int j = 0; j < inputSize; j++) {
            double grad = 0.0;
            for (int i = 0; i < outputSize; i++) {
                grad += layerGradient[i] * weights[i][j];
            }
            inputGradient[j] = grad;
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightGradients[i][j] = layerGradient[i] * lastInputs[j];
            }
            biasGradients[i] = layerGradient[i];
        }
        
        return new GradientPackage(weightGradients, biasGradients, inputGradient);
    }

    @Override
    public void applyGradients(GradientPackage gradients, double learningRate) {
        double[][] weightGradients = gradients.getWeightGradients();
        double[] biasGradients = gradients.getBiasGradients();

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] -= learningRate * weightGradients[i][j];
            }
            
            biases[i] -= learningRate * biasGradients[i];
        }
    }
    
    @Override
    public GradientPackage createEmptyGradients() {
        double[][] weightGradients = new double[outputSize][inputSize];
        double[] biasGradients = new double[outputSize];
        double[] inputGradient = new double[inputSize]; 
        
        return new GradientPackage(weightGradients, biasGradients, inputGradient);
    }

    @Override
    public int getOutputSize() {
        return outputSize;
    }
}