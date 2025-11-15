package ifndef.nn.util;

import java.io.Serializable;

public class DataNormalizer implements Serializable {

    private static final long serialVersionUID = 1L;

    private final double dataMin;
    private final double dataMax;

    private final double normalizedMin;
    private final double normalizedMax;

    public DataNormalizer(double dataMin, double dataMax) {
        this.dataMin = dataMin;
        this.dataMax = dataMax;
        this.normalizedMin = -1.0;
        this.normalizedMax = 1.0;
    }

    public double normalize(double value) {
        if (dataMax - dataMin == 0) {
            return (normalizedMin + normalizedMax) / 2.0;
        }
        
        return normalizedMin + (value - dataMin) * (normalizedMax - normalizedMin) / (dataMax - dataMin);
    }

    public double denormalize(double value) {
        if (normalizedMax - normalizedMin == 0) {
            return dataMin;
        }
        
        return dataMin + (value - normalizedMin) * (dataMax - dataMin) / (normalizedMax - normalizedMin);
    }
}