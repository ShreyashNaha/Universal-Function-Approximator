package ifndef.nn.training;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class Dataset implements Serializable {

    private static final long serialVersionUID = 1L;

    private final List<DataPoint> data;
    
    private transient Function<Double, Double> targetFunction;

    private double xMin = Double.POSITIVE_INFINITY;
    private double xMax = Double.NEGATIVE_INFINITY;
    private double yMin = Double.POSITIVE_INFINITY;
    private double yMax = Double.NEGATIVE_INFINITY;

    public Dataset(Function<Double, Double> targetFunction) {
        this.data = new ArrayList<>();
        this.targetFunction = targetFunction;
    }
    
    public Dataset() {
        this.data = new ArrayList<>();
        this.targetFunction = null;
    }

    public void generateData(int numSamples, double minX, double maxX) {
        this.data.clear();
        this.xMin = minX;
        this.xMax = maxX;
        this.yMin = Double.POSITIVE_INFINITY;
        this.yMax = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < numSamples; i++) {
            double x = minX + (Math.random() * (maxX - minX));
            double y = this.targetFunction.apply(x);

            if (Double.isNaN(y) || Double.isInfinite(y)) {
                i--;
                continue;
            }

            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;

            this.data.add(new DataPoint(new double[]{x}, new double[]{y}));
        }
    }

    public void add(DataPoint dp) {
        this.data.add(dp);
    }

    public void shuffle() {
        Collections.shuffle(this.data);
    }

    public List<DataPoint> getData() {
        return this.data;
    }

    public int size() {
        return this.data.size();
    }
    
    public double getXMin() { return xMin; }
    public double getXMax() { return xMax; }
    public double getYMin() { return yMin; }
    public double getYMax() { return yMax; }
}