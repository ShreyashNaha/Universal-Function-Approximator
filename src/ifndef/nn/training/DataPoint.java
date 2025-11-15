package ifndef.nn.training;

import java.io.Serializable;

public class DataPoint implements Serializable {

    private static final long serialVersionUID = 1L;

    private final double[] inputs;
    private final double[] targets;

    public DataPoint(double[] inputs, double[] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public double[] getInputs() {
        return inputs;
    }

    public double[] getTargets() {
        return targets;
    }
}