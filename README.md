# Neural Network Java Library

This repository contains a custom Java neural network library, including activation functions, core neural network components, loss functions, training utilities, and helper tools. Designed for both simple and complex function approximation.

---

## File Structure

src/
└── ifndef/
    └── nn/
        ├── activation/
        │   ├── ActivationFunction.java
        │   ├── ReLU.java
        │   ├── Sigmoid.java
        │   └── Tanh.java
        │
        ├── core/
        │   ├── DenseLayer.java
        │   ├── GradientPackage.java
        │   ├── Layer.java
        │   └── NeuralNetwork.java
        │
        ├── loss/
        │   ├── LossFunction.java
        │   └── MeanSquaredError.java
        │
        ├── training/
        │   ├── BatchTask.java
        │   ├── BatchTaskResult.java
        │   ├── DataPoint.java
        │   ├── Dataset.java
        │   └── Trainer.java
        │
        ├── util/
        │   ├── DataNormalizer.java
        │   ├── FunctionInfo.java
        │   ├── GraphingUtil.java
        │   ├── NetworkSerializer.java
        │   └── TrainedModel.java
        │
        └── Main.java


---

## Core Functionality Summary

### Two-Track Training System 🐞

- **Track 1: Raw Training (No Normalization)**  
  - Use case: Simple functions like `2x`, `x^2`, `0.5x + 3`.  
  - Advantage: Allows the network to extrapolate to large values (e.g., `f(50)` or `f(10000)`).

- **Track 2: Normalized Training**  
  - Use case: Complex functions like `log`, `sin`, `e^x`.  
  - Advantage: Scales all data to [-1, 1] for accurate learning of complex shapes.  
  - Limitation: Cannot extrapolate outside the training range.

---

### Multithreaded Trainer 🚀

- Uses all CPU cores to process mini-batches in parallel.  
- Faster than single-sample training, especially for deep networks and large batch sizes.  
- If training feels slow, consider increasing batch size to reduce thread overhead.

---

## Recommended Settings by Function

| Function        | Architecture | Activation | Epochs        | Samples         | Batch Size    | Rationale / Why? |
|-----------------|-------------|-----------|---------------|----------------|---------------|-----------------|
| f(x) = 2x / 0.5x+3 | Standard    | ReLU      | 500 - 1,000   | 2,000 - 5,000  | 32 or 64     | Simple linear. Deep network overkill. ReLU trains fast and extrapolates well. |
| f(x) = x^2      | Standard    | ReLU      | 2,000 - 4,000 | ~10,000        | 64            | Simple non-linear. ReLU works for "U" shape. Standard network is enough. Will extrapolate. |
| f(x) = log(x)   | Deep        | ReLU      | 3,000 - 5,000 | 20,000+        | 128           | Complex curve. Deep network required. Accurate inside [0.1, 10], fails outside. |
| f(x) = sin(x)   | Deep        | Tanh      | 5,000 - 8,000 | 50,000 - 100,000 | 128 or 256  | Periodic function. Tanh builds waves, Deep network needed. ReLU fails. |
| f(x) = e^x      | Deep        | ReLU      | 4,000 - 6,000 | 25,000+        | 128 or 256   | Explosive growth. Deep network needed. ReLU is unbounded, models rapid growth within normalized range. |

---

## Parameter Trade-offs Explained

### Architecture (Standard vs Deep)
- **Standard (2 layers, 16 neurons)**: Fast, simple, good for linear/non-linear functions.  
- **Deep (4 layers, 32 neurons)**: Slower, powerful, needed for complex functions like `sin` or `log`.

### Activation (ReLU vs Tanh)
- **ReLU (max(0, x))**: Default choice, works for monotonically increasing functions (`x^2`, `log`, `e^x`, `2x`).  
- **Tanh (-1 to 1)**: Use for `sin(x)`. Natural wave-shape, easy for network to combine into periodic patterns.

### Epochs vs Samples
- **Samples**: Quality of the dataset ("textbook").  
- **Epochs**: Number of passes over the data ("reading the book").  
- **Advice**: More samples are better than more epochs. If loss stagnates, increase samples first.

### Batch Size (Speed/Accuracy Knob ⚙️)
- **Small (32)**: Slower, more accurate ("noisy" updates help optimization).  
- **Large (128, 256)**: Faster (benefits multithreading), slightly less accurate.  
- **Recommendation**: Start with 128 or 256 for good speed and accuracy balance.

---

## Getting Started

1. Clone the repository:
2. In project folder, open terminal and run this command in order to compile : javac -d bin -sourcepath src src/ifndef/nn/Main.java
3. java -cp bin ifndef.nn.Main write this afterwards in order to execute the code.



