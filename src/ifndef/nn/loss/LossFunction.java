package ifndef.nn.loss;

import java.io.Serializable;

public interface LossFunction extends Serializable {

    long serialVersionUID = 1L;

    double compute(double[] predictions, double[] targets);

    double[] derivative(double[] predictions, double[] targets);
}