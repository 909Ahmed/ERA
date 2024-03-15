# Session 7 Assignment

Goal - Achieve 99.4 accuracy on MNIST dataset, under 8k parameters and 15 epochs.

### Model 1

[Model1](./model1.ipynb)

### Target

- Have a basic working code
- Have a skeleton code to build upon
- Make it as light as possible without adding any normalisation, regularisation, etc.

### Results

- EPOCHS : 20
- Parameters: 10.7k
- Best Train Accuracy: 99.19
- Best Test Accuracy: 98.82

### Analysis

- Working
- Overfitting
- Need regularisation

---

### Model 2

[Model2](./model2.ipynb)

### Target

- Add normalisation, regularisation, GAP.
- Use transition blocks
- Keep the parameters same

### Results

- Parameters: 10.8k
- Best Train Accuracy: 99.89
- Best Test Accuracy: 99.44

### Analysis

- Normalisatoin, Dropout, GAP working.
- No overfitting
- The model is not over-fitting at all. 

---

### Model 3

[Model3](./model3.ipynb)

### Target

- Add image augmentation.
- Properly use transition blocks
- Used Adam with weight decay.

### Results

- Parameters: 8k
- Best Train accuracy: 98.73
- Best Test accuracy: 99.40

### Analysis

- Achieved high accuracy faster, but stabilised around 99.3
- Can achieve more accuracy if trained more.
- No overfitting
- High potent model even  with less parameters

## Results

We have achieved our goal of achieving 99.4 (consistent at the end) accuracy on MNIST using [Model3](./model3.ipynb).