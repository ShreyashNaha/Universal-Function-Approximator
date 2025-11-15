package ifndef.nn.core;

import ifndef.nn.loss.LossFunction;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork implements Serializable {

    private static final long serialVersionUID = 1L; 
    private final List<Layer> layers = new ArrayList<>();

    public void add(Layer layer) {
        this.layers.add(layer);
    }

    public double[] predict(double[] inputs) {
        double[] currentOutput = inputs;
        for (Layer layer : layers) {
            currentOutput = layer.forward(currentOutput);
        }
        return currentOutput;
    }

    public List<GradientPackage> backward(double[] lossGradient) {
        List<GradientPackage> allGradients = new ArrayList<>();
        double[] currentGradient = lossGradient;

        for (int i = layers.size() - 1; i >= 0; i--) {
            GradientPackage layerGrads = layers.get(i).backward(currentGradient);
            allGradients.add(0, layerGrads); // Add in forward order
            currentGradient = layerGrads.getInputGradient();
        }
        
        return allGradients;
    }

    public void applyGradients(List<GradientPackage> averagedGradients, double learningRate) {
        if (averagedGradients.size() != layers.size()) {
            throw new IllegalArgumentException("Gradient list size must match layer count.");
        }
        
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).applyGradients(averagedGradients.get(i), learningRate);
        }
    }

    public List<GradientPackage> createEmptyGradientsList() {
        List<GradientPackage> list = new ArrayList<>();
        for (Layer layer : layers) {
            list.add(layer.createEmptyGradients());
        }
        return list;
    }
}