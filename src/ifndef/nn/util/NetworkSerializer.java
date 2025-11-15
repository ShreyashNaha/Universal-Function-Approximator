package ifndef.nn.util;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class NetworkSerializer {

    public static void save(TrainedModel model, String filename) {
        try (FileOutputStream fos = new FileOutputStream(filename);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            
            oos.writeObject(model);
            System.out.println("Model successfully saved to " + filename);

        } catch (IOException e) {
            System.err.println("Error saving model: " + e.getMessage());
        }
    }

    public static TrainedModel load(String filename) {
        TrainedModel model = null;
        try (FileInputStream fis = new FileInputStream(filename);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            
            model = (TrainedModel) ois.readObject();
            System.out.println("Model successfully loaded from " + filename);

        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading model: " + e.getMessage());
        }
        return model;
    }
}