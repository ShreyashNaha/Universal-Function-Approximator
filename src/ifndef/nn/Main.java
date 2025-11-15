package ifndef.nn;

import ifndef.nn.activation.ActivationFunction;
import ifndef.nn.activation.ReLU;
import ifndef.nn.activation.Tanh;
import ifndef.nn.core.DenseLayer;
import ifndef.nn.core.NeuralNetwork;
import ifndef.nn.loss.LossFunction;
import ifndef.nn.loss.MeanSquaredError;
import ifndef.nn.training.DataPoint;
import ifndef.nn.training.Dataset;
import ifndef.nn.training.Trainer;
import ifndef.nn.util.DataNormalizer;
import ifndef.nn.util.FunctionInfo;
import ifndef.nn.util.GraphingUtil;
import ifndef.nn.util.NetworkSerializer;
import ifndef.nn.util.TrainedModel;

import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;

public class Main {

    private static final Scanner scanner = new Scanner(System.in);
    private static Trainer trainer; // Hold the trainer to shut it down
    
    private static NeuralNetwork network;
    private static FunctionInfo currentFunction;
    private static DataNormalizer inputNormalizer;  // Null if not normalized
    private static DataNormalizer outputNormalizer; // Null if not normalized

    public static void main(String[] args) {
        while (true) {
            showMainMenu();
            int choice = getUserInput(1, 3);
            
            switch (choice) {
                case 1:
                    trainNewNetwork();
                    break;
                case 2:
                    loadNetwork();
                    break;
                case 3:
                    System.out.println("Exiting. Goodbye!");
                    if (trainer != null) {
                        trainer.shutdown(); // Shut down the thread pool
                    }
                    scanner.close();
                    return;
            }
            
            if (network != null) {
                runPostTrainingMenu();
            }
        }
    }

    private static void showMainMenu() {
        System.out.println("\n--- Neural Network Main Menu ---");
        System.out.println("1. Train a New Network");
        System.out.println("2. Load a Network");
        System.out.println("3. Exit");
        System.out.print("Please select an option: ");
    }

    private static void trainNewNetwork() {
        selectFunction();

        network = selectArchitecture();

        int epochs = getDynamicInput("\nEnter number of epochs (1 - 100,000): ", 1, 100000);
        int numSamples = getDynamicInput("\nEnter number of samples (1 - 100,000): ", 1, 100000);
        int batchSize = getDynamicInput("\nEnter batch size (e.g., 32, 64): ", 1, numSamples);
        double learningRate = 0.001;
        
        System.out.print("\nStart training? (y/n): ");
        if (!scanner.next().equalsIgnoreCase("y")) {
            System.out.println("Training cancelled.");
            clearModelFromMemory();
            return;
        }
        
        Dataset rawDataset = new Dataset(currentFunction.function);
        rawDataset.generateData(numSamples, currentFunction.minX, currentFunction.maxX);
        
        Dataset trainingDataset; 
        
        if (currentFunction.shouldNormalize) {
            System.out.println("Generating and normalizing data...");
            inputNormalizer = new DataNormalizer(rawDataset.getXMin(), rawDataset.getXMax());
            outputNormalizer = new DataNormalizer(rawDataset.getYMin(), rawDataset.getYMax());
            
            trainingDataset = new Dataset();
            for (DataPoint dp : rawDataset.getData()) {
                double normalizedX = inputNormalizer.normalize(dp.getInputs()[0]);
                double normalizedY = outputNormalizer.normalize(dp.getTargets()[0]);
                trainingDataset.add(new DataPoint(new double[]{normalizedX}, new double[]{normalizedY}));
            }
        } else {
            System.out.println("Generating data (Raw, no normalization)...");
            trainingDataset = rawDataset;
            inputNormalizer = null;  
            outputNormalizer = null;
        }

        LossFunction loss = new MeanSquaredError();
        
        if (trainer != null) {
            trainer.shutdown();
        }
        trainer = new Trainer(network, loss, learningRate);

        System.out.println("Starting training on " + currentFunction.name + "...");
        int printInterval = Math.max(1, epochs / 10); 
        
        long startTime = System.nanoTime();
        
        trainer.train(trainingDataset, epochs, batchSize, printInterval);
        
        long endTime = System.nanoTime();
        double timeElapsedSeconds = (endTime - startTime) / 1_000_000_000.0;
        System.out.printf("Total training time: %.3f seconds\n", timeElapsedSeconds);
        
        System.out.print("\nSave this model? (y/n): ");
        if (scanner.next().equalsIgnoreCase("y")) {
            System.out.print("Enter filename to save (e.g., model.dat): ");
            String filename = scanner.next();
            
            TrainedModel model = new TrainedModel(network, currentFunction, inputNormalizer, outputNormalizer);
            NetworkSerializer.save(model, filename);
        }
    }
    
    private static void selectFunction() {
        System.out.println("\nPlease choose a function to learn:");
        System.out.println("1. f(x) = 2x           (Linear)");
        System.out.println("2. f(x) = x^2          (Quadratic)");
        System.out.println("3. f(x) = log(x)       (Logarithmic, Normalized)");
        System.out.println("4. f(x) = sin(x)       (Periodic, Normalized)");
        System.out.println("5. f(x) = e^x          (Exponential, Normalized)");
        System.out.println("6. f(x) = 0.5x + 3     (Linear w/ Bias)");
        System.out.print("Select (1-6): ");

        int choice = getUserInput(1, 6);
        currentFunction = getFunctionInfoFromID(choice);
    }

    private static ActivationFunction selectActivation() {
        System.out.println("\nPlease choose an activation function for hidden layers:");
        System.out.println("1. ReLU (Good for most functions)");
        System.out.println("2. Tanh (Excellent for periodic functions like sin(x))");
        System.out.print("Select (1-2): ");
        
        int choice = getUserInput(1, 2);
        return (choice == 1) ? new ReLU() : new Tanh();
    }
    
    private static NeuralNetwork selectArchitecture() {
        System.out.println("\nPlease choose a network architecture:");
        System.out.println("1. Standard (2 hidden layers, 16 neurons each)");
        System.out.println("2. Deep (4 hidden layers, 32 neurons each)");
        System.out.print("Select (1-2): ");
        
        int archChoice = getUserInput(1, 2);
        ActivationFunction activation = selectActivation();
        NeuralNetwork nn = new NeuralNetwork();
        
        if (archChoice == 1) {
            nn.add(new DenseLayer(1, 16, activation));
            nn.add(new DenseLayer(16, 16, activation));
            nn.add(new DenseLayer(16, 1, null)); // Linear output layer
        } else {
            nn.add(new DenseLayer(1, 32, activation));
            nn.add(new DenseLayer(32, 32, activation));
            nn.add(new DenseLayer(32, 32, activation));
            nn.add(new DenseLayer(32, 32, activation));
            nn.add(new DenseLayer(32, 1, null)); // Linear output layer
        }
        return nn;
    }

    private static void loadNetwork() {
        System.out.print("\nEnter filename to load (e.g., model.dat): ");
        String filename = scanner.next();
        TrainedModel model = NetworkSerializer.load(filename);
        
        if (model == null) {
            System.out.println("Failed to load model.");
            clearModelFromMemory();
        } else {
            network = model.getNetwork();
            currentFunction = getFunctionInfoFromName(model.getFunctionInfo().name);
            inputNormalizer = model.getInputNormalizer();
            outputNormalizer = model.getOutputNormalizer();
        }
    }

    private static void runPostTrainingMenu() {
        while (network != null) {
            System.out.println("\n--- Model Menu (" + currentFunction.name + ") ---");
            System.out.println("1. Interactively Test");
            System.out.println("2. Generate Performance Graph");
            System.out.println("3. Return to Main Menu");
            System.out.print("Please select an option: ");

            int choice = getUserInput(1, 3);
            switch (choice) {
                case 1:
                    runTestingLoop();
                    break;
                case 2:
                    generateGraph();
                    break;
                case 3:
                    clearModelFromMemory();
                    return;
            }
        }
    }

    private static void runTestingLoop() {
        System.out.println("\n--- Interactive Testing ---");
        System.out.println("Enter a number to predict, or 'q' to return.");

        while (true) {
            System.out.print("\nTest Input > ");
            String input = scanner.next();

            if (input.equalsIgnoreCase("q")) {
                break; 
            }

            try {
                double x = Double.parseDouble(input);
                double prediction;
                
                if (currentFunction.shouldNormalize) {
                    double normalizedX = inputNormalizer.normalize(x);
                    double[] normalizedPrediction = network.predict(new double[]{normalizedX});
                    prediction = outputNormalizer.denormalize(normalizedPrediction[0]);
                } else {
                    prediction = network.predict(new double[]{x})[0];
                }
                
                System.out.printf("Network Prediction: %.6f\n", prediction);
                
                double expected = currentFunction.function.apply(x);
                System.out.printf("Expected Value:     %.6f\n", expected);

                if (currentFunction.shouldNormalize && (x < currentFunction.minX || x > currentFunction.maxX)) {
                    System.out.println("(Warning: Input is outside the trained range, prediction may be unreliable)");
                }

            } catch (NumberFormatException e) {
                System.out.println("Invalid input. Please enter a number or 'q'.");
            }
        }
    }

    private static void generateGraph() {
        System.out.println("\nGenerating graph data...");
        int points = 1000; // Use 1000 points for a smooth graph
        List<Double> xValues = new ArrayList<>();
        List<Double> expected = new ArrayList<>();
        List<Double> predicted = new ArrayList<>();

        double step = (currentFunction.maxX - currentFunction.minX) / (points - 1);
        for (int i = 0; i < points; i++) {
            double x = currentFunction.minX + (i * step);
            double yTrue = currentFunction.function.apply(x);
            double yPred;

            if (currentFunction.shouldNormalize) {
                double normalizedX = inputNormalizer.normalize(x);
                double[] normalizedPred = network.predict(new double[]{normalizedX});
                yPred = outputNormalizer.denormalize(normalizedPred[0]);
            } else {
                yPred = network.predict(new double[]{x})[0];
            }

            xValues.add(x);
            expected.add(yTrue);
            predicted.add(yPred);
        }

        String title = "Performance on: " + currentFunction.name;
        GraphingUtil.plot(xValues, expected, predicted, title);
    }

    private static void clearModelFromMemory() {
        network = null;
        currentFunction = null;
        inputNormalizer = null;
        outputNormalizer = null;
    }

    private static int getUserInput(int min, int max) {
        int choice = -1;
        while (choice < min || choice > max) {
            try {
                choice = scanner.nextInt();
                if (choice < min || choice > max) {
                    System.out.print("Invalid choice. Please enter a number between " + min + " and " + max + ": ");
                }
            } catch (InputMismatchException e) {
                System.out.print("Invalid input. Please enter a number: ");
                scanner.next(); // Clear the bad input
            }
        }
        return choice;
    }

    private static int getDynamicInput(String prompt, int min, int max) {
        System.out.print(prompt);
        return getUserInput(min, max);
    }

    private static FunctionInfo getFunctionInfoFromID(int choice) {
        switch (choice) {
            case 1: return new FunctionInfo("f(x) = 2x", x -> 2.0 * x, -10.0, 10.0, false);
            case 2: return new FunctionInfo("f(x) = x^2", x -> x * x, -5.0, 5.0, false);
            case 6: return new FunctionInfo("f(x) = 0.5x + 3", x -> 0.5 * x + 3.0, -20.0, 20.0, false);
            case 3: return new FunctionInfo("f(x) = log(x)", Math::log, 0.1, 10.0, true);
            case 4: return new FunctionInfo("f(x) = sin(x)", x -> Math.sin(Math.toRadians(x)), -720.0, 720.0, true);
            case 5: return new FunctionInfo("f(x) = e^x", Math::exp, 0.0, 7.0, true); 
            
            default: throw new IllegalArgumentException("Invalid choice");
        }
    }

    private static FunctionInfo getFunctionInfoFromName(String name) {
        switch (name) {
            case "f(x) = 2x":       return getFunctionInfoFromID(1);
            case "f(x) = x^2":      return getFunctionInfoFromID(2);
            case "f(x) = log(x)":     return getFunctionInfoFromID(3);
            case "f(x) = sin(x)":   return getFunctionInfoFromID(4);
            case "f(x) = e^x":      return getFunctionInfoFromID(5);
            case "f(x) = 0.5x + 3": return getFunctionInfoFromID(6);
            default: 
                return new FunctionInfo("Unknown Function", x -> Double.NaN, -10, 10, false);
        }
    }
}