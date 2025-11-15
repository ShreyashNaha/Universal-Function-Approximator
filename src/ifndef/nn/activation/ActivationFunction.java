package ifndef.nn.activation;

import java.io.Serializable;

public interface ActivationFunction extends Serializable {

    double value(double x);

    double derivative(double x);
}