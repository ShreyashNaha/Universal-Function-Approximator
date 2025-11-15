package ifndef.nn.activation;


public class Sigmoid implements ActivationFunction {

    @Override
    public double value(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        // Derivative is f'(x) = f(x) * (1 - f(x))
        double sig = value(x);
        return sig * (1.0 - sig);
    }
}