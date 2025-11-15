package ifndef.nn.activation;

public class Tanh implements ActivationFunction {

    @Override
    public double value(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        // Derivative is f'(x) = 1 - (tanh(x))^2
        double tanh_x = Math.tanh(x);
        return 1.0 - (tanh_x * tanh_x);
    }
}