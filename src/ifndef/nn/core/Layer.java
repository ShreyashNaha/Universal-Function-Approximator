package ifndef.nn.core;

import java.io.Serializable;

public interface Layer extends Serializable {

    double[] forward(double[] inputs);

    GradientPackage backward(double[] outputGradient);

    void applyGradients(GradientPackage gradients, double learningRate);

    GradientPackage createEmptyGradients();

    int getOutputSize();
}