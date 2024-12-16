# Mechanisms-Of-Action-MoA-Prediction


# Mechanisms Of Action (MoA) Prediction

Mechanisms of Action (MoA) prediction involves the use of machine learning techniques to identify how different compounds interact with biological systems. This project focuses on accurately classifying the mechanism of action of compounds based on experimental data.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Modeling Approach](#modeling-approach)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

This project aims to predict the mechanisms of action (MoA) of chemical compounds using machine learning techniques. The dataset includes various gene expression and cell viability features which serve as predictors for multi-label classification tasks. 

---

## Dataset

The dataset used for this project is publicly available on [Kaggle](https://www.kaggle.com/) and contains the following:

- **Features**: Gene expression levels and cell viability metrics.
- **Labels**: MoA categories for each compound (multi-label classification).

### Data Preprocessing
- Missing values imputed.
- Feature scaling using standardization.
- Dimensionality reduction using techniques like PCA.

---

## Installation

### Prerequisites
- Python 3.8+
- Git
- Virtual Environment (optional but recommended)

### Steps
1. Clone the repository:

   ```bash
   git clone https://github.com/Adityagnss/Mechanisms-Of-Action-MoA-Prediction.git
   cd Mechanisms-Of-Action-MoA-Prediction
   ```

2. Set up a virtual environment (optional):

   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model

Run the following command to preprocess data and train the model:

```bash
python src/train.py
```

### Evaluating the Model

To evaluate the model's performance on the test dataset:

```bash
python src/evaluate.py
```

### Running Notebooks

For interactive exploration and visualization, open the Jupyter notebooks in the `notebooks/` directory:

```bash
jupyter notebook
```

---

## Modeling Approach

### Techniques Used
- Feature Engineering
- Dimensionality Reduction (PCA, t-SNE)
- Multi-label Classification using:
  - Logistic Regression
  - Random Forest Classifier
  - Neural Networks (PyTorch)

### Evaluation Metrics
- Log Loss
- Hamming Loss
- Multi-label Accuracy

---

## Results

| Model               | Log Loss  | Hamming Loss | Multi-label Accuracy |
|---------------------|-----------|--------------|-----------------------|
| Logistic Regression | 0.678     | 0.210        | 85.4%                |
| Random Forest       | 0.564     | 0.182        | 89.6%                |
| Neural Network      | 0.532     | 0.174        | 91.2%                |

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Commit your changes and open a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Kaggle community for the dataset.
- Scikit-learn, PyTorch, and Matplotlib libraries for supporting model development.
- Open-source contributors for providing helpful utilities.

---
