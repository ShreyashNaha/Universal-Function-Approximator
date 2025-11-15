package ifndef.nn.loss;

public class MeanSquaredError implements LossFunction {

    private static final long serialVersionUID = 1L;

    @Override
    public double compute(double[] predictions, double[] targets) {
        if (predictions.length != targets.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same length.");
        }

        double sumOfSquares = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double error = predictions[i] - targets[i];
            sumOfSquares += (error * error);
        }

        return sumOfSquares / predictions.length;
    }

    @Override
    public double[] derivative(double[] predictions, double[] targets) {
        if (predictions.length != targets.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same length.");
        }

        double[] gradient = new double[predictions.length];
        double n = predictions.length;

        for (int i = 0; i < predictions.length; i++) {
            // Derivative of MSE is 2 * (pred - target) / n
            gradient[i] = (2.0 / n) * (predictions[i] - targets[i]);
        }

        return gradient;
    }
}