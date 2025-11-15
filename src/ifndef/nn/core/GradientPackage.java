package ifndef.nn.core;


public class GradientPackage {

    private final double[][] weightGradients;
    private final double[] biasGradients;
    private final double[] inputGradient; 

    public GradientPackage(double[][] weightGradients, double[] biasGradients, double[] inputGradient) {
        this.weightGradients = weightGradients;
        this.biasGradients = biasGradients;
        this.inputGradient = inputGradient;
    }

    public double[][] getWeightGradients() {
        return weightGradients;
    }

    public double[] getBiasGradients() {
        return biasGradients;
    }

    public double[] getInputGradient() {
        return inputGradient;
    }
    
    public void add(GradientPackage other) {
        for (int i = 0; i < this.weightGradients.length; i++) {
            for (int j = 0; j < this.weightGradients[i].length; j++) {
                this.weightGradients[i][j] += other.weightGradients[i][j];
            }
        }
        
        for (int i = 0; i < this.biasGradients.length; i++) {
            this.biasGradients[i] += other.biasGradients[i];
        }
    }
}