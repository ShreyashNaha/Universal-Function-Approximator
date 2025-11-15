package ifndef.nn.training;

import ifndef.nn.core.GradientPackage;
import ifndef.nn.core.NeuralNetwork;
import ifndef.nn.loss.LossFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Trainer {

    private final NeuralNetwork network;
    private final LossFunction lossFunction;
    private final double learningRate;
    private final int numThreads;
    private final ExecutorService threadPool;

    public Trainer(NeuralNetwork network, LossFunction lossFunction, double learningRate) {
        this.network = network;
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;

        this.numThreads = Runtime.getRuntime().availableProcessors();
        this.threadPool = Executors.newFixedThreadPool(this.numThreads);
        System.out.println("Trainer initialized with " + this.numThreads + " threads.");
    }

    public void train(Dataset dataset, int epochs, int batchSize, int printInterval) {
        
        List<DataPoint> data = dataset.getData();
        int numBatches = (int) Math.ceil((double) data.size() / batchSize);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            dataset.shuffle();
            double totalLoss = 0.0;

            for (int b = 0; b < numBatches; b++) {

                int from = b * batchSize;
                int to = Math.min(from + batchSize, data.size());
                List<DataPoint> miniBatch = data.subList(from, to);
                int miniBatchSize = miniBatch.size();

                try {
                    List<GradientPackage> totalGradients = network.createEmptyGradientsList();

                    List<Future<BatchTaskResult>> futures = new ArrayList<>();
                    int subBatchSize = (int) Math.ceil((double) miniBatchSize / numThreads);

                    for (int t = 0; t < numThreads; t++) {
                        int tFrom = t * subBatchSize;
                        int tTo = Math.min(tFrom + subBatchSize, miniBatchSize);
                        if (tFrom < tTo) {
                            List<DataPoint> subBatch = miniBatch.subList(tFrom, tTo);
                            BatchTask task = new BatchTask(network, lossFunction, subBatch);
                            futures.add(threadPool.submit(task));
                        }
                    }

                    double batchLoss = 0.0;
                    for (Future<BatchTaskResult> future : futures) {
                        BatchTaskResult result = future.get(); 
                        batchLoss += result.getBatchLoss();
                        List<GradientPackage> threadGradients = result.getGradients();

                        for (int i = 0; i < totalGradients.size(); i++) {
                            totalGradients.get(i).add(threadGradients.get(i));
                        }
                    }

                    for (GradientPackage grad : totalGradients) {
                        
                        for (double[] row : grad.getWeightGradients()) {
                            for (int j = 0; j < row.length; j++) {
                                row[j] /= miniBatchSize;
                            }
                        }
                        
                        for (int i = 0; i < grad.getBiasGradients().length; i++) {
                            grad.getBiasGradients()[i] /= miniBatchSize;
                        }
                    }
                    
                    network.applyGradients(totalGradients, learningRate);
                    
                    totalLoss += batchLoss;

                } catch (Exception e) {
                    System.err.println("Error during multi-threaded training: " + e.getMessage());
                    e.printStackTrace();
                }
            }

            if (epoch % printInterval == 0 || epoch == epochs) {
                double averageLoss = totalLoss / data.size();
                System.out.printf("Epoch: %d/%d, Average Loss: %.8f\n", epoch, epochs, averageLoss);
            }
        }
        
        System.out.println("Training finished.");
    }
    
    public void shutdown() {
        this.threadPool.shutdown();
    }
}