package ifndef.nn.training;

import ifndef.nn.core.GradientPackage;
import java.util.List;

public class BatchTaskResult {

    private final List<GradientPackage> gradients;
    private final double batchLoss;

    public BatchTaskResult(List<GradientPackage> gradients, double batchLoss) {
        this.gradients = gradients;
        this.batchLoss = batchLoss;
    }

    public List<GradientPackage> getGradients() {
        return gradients;
    }

    public double getBatchLoss() {
        return batchLoss;
    }
}