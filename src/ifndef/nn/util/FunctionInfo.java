package ifndef.nn.util;

import java.io.Serializable;
import java.util.function.Function;

public class FunctionInfo implements Serializable {

    private static final long serialVersionUID = 1L;

    public final String name;
    public final double minX;
    public final double maxX;
    
    public final boolean shouldNormalize;
    
    public transient Function<Double, Double> function;

    public FunctionInfo(String name, Function<Double, Double> function, 
                        double minX, double maxX, boolean shouldNormalize) {
        this.name = name;
        this.function = function;
        this.minX = minX;
        this.maxX = maxX;
        this.shouldNormalize = shouldNormalize;
    }
}