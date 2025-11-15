package ifndef.nn.training;

import ifndef.nn.core.GradientPackage;
import ifndef.nn.core.NeuralNetwork;
import ifndef.nn.loss.LossFunction;

import java.util.List;
import java.util.concurrent.Callable;

public class BatchTask implements Callable<BatchTaskResult> {

    private final NeuralNetwork network;
    private final LossFunction lossFunction;
    private final List<DataPoint> subBatch;

    public BatchTask(NeuralNetwork network, LossFunction lossFunction, List<DataPoint> subBatch) {
        this.network = network;
        this.lossFunction = lossFunction;
        this.subBatch = subBatch;
    }

    @Override
    public BatchTaskResult call() throws Exception {

        List<GradientPackage> totalGradients = network.createEmptyGradientsList();
        double totalLoss = 0.0;

        for (DataPoint dp : subBatch) {
            double[] predictions = network.predict(dp.getInputs());

            totalLoss += lossFunction.compute(predictions, dp.getTargets());
            double[] lossGradient = lossFunction.derivative(predictions, dp.getTargets());

            List<GradientPackage> sampleGradients = network.backward(lossGradient);

            for (int i = 0; i < totalGradients.size(); i++) {
                totalGradients.get(i).add(sampleGradients.get(i));
            }
        }

        return new BatchTaskResult(totalGradients, totalLoss);
    }
}