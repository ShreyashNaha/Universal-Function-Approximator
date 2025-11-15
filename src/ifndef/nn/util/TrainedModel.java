package ifndef.nn.util;

import ifndef.nn.core.NeuralNetwork;
import java.io.Serializable;

public class TrainedModel implements Serializable {

    private static final long serialVersionUID = 2L; 

    private final NeuralNetwork network;
    private final FunctionInfo functionInfo;
    
    private final DataNormalizer inputNormalizer;
    private final DataNormalizer outputNormalizer;

    public TrainedModel(NeuralNetwork network, FunctionInfo functionInfo,
                        DataNormalizer inputNormalizer, DataNormalizer outputNormalizer) {
        this.network = network;
        this.functionInfo = functionInfo;
        this.inputNormalizer = inputNormalizer;
        this.outputNormalizer = outputNormalizer;
    }

    public NeuralNetwork getNetwork() {
        return network;
    }

    public FunctionInfo getFunctionInfo() {
        return functionInfo;
    }

    public DataNormalizer getInputNormalizer() {
        return inputNormalizer;
    }

    public DataNormalizer getOutputNormalizer() {
        return outputNormalizer;
    }
}