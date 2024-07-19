# Customer Churn Prediction Model

## Overview

This project evaluates different machine learning models for customer churn prediction. The models compared are Logistic Regression, Random Forest, and Support Vector Machine (SVM). The evaluation includes accuracy, precision, recall, F1-score, and confusion matrices to determine the best-performing model for predicting customer churn.

## Dependencies

The project requires the following Python packages:

- `matplotlib` for plotting
- `seaborn` for advanced visualizations
- `numpy` for numerical operations

You can install the required packages using pip:

```bash
pip install matplotlib seaborn numpy
```

## Code Description

The provided code performs the following tasks:

1. **Model Names and Metrics**:
   - Defines the names of the models and their corresponding accuracy scores.
   - Contains precision, recall, and F1-score for each class (0 and 1) for all models.
   - Includes confusion matrices for each model.

2. **Visualizations**:
   - **Accuracy Comparison**: A bar plot showing the accuracy of each model.
   - **Confusion Matrices**: Heatmaps displaying confusion matrices for each model.
   - **Precision, Recall, and F1-Score Comparison**: Bar plots comparing these metrics across models and classes.

### Accuracy Comparison

The accuracy comparison plot provides a visual representation of the overall accuracy of each model. It helps to quickly assess which model performs best in terms of overall correctness.

```python
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0, 1)
plt.show()
```

### Confusion Matrices

Confusion matrices for each model are visualized using heatmaps. These matrices help in understanding the distribution of true positives, false positives, true negatives, and false negatives for each model.

```python
for model in models:
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrices[model], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model} - Confusion Matrix')
    plt.show()
```

### Precision, Recall, and F1-Score Comparison

The bar plots for precision, recall, and F1-score allow for a detailed comparison of how well each model performs for each class (0 and 1). This is useful for evaluating performance beyond just accuracy, particularly in cases of class imbalance.

```python
metric_names = ['precision', 'recall', 'f1-score']
for metric in metric_names:
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    index = np.arange(2)  # Two classes: 0 and 1
    for i, model in enumerate(models):
        plt.bar(index + i * bar_width, metrics[model][metric], bar_width, label=model)

    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.title(f'Comparison of {metric.capitalize()}')
    plt.xticks(index + bar_width, ['No Churn', 'Churn'])
    plt.legend()
    plt.ylim(0, 1)
    plt.show()
```

## Usage

To run the script and generate the visualizations, ensure that you have the necessary dependencies installed and run the Python script:

```bash
python evaluate_models.py
```

### Output

The script will produce the following outputs:

- **Accuracy Comparison**: A bar plot comparing the accuracy of different models.
- **Confusion Matrices**: Heatmaps displaying the confusion matrices for each model.
- **Precision, Recall, and F1-Score Comparison**: Bar plots comparing these metrics for each model and class.

## Notes

- Ensure that the necessary libraries are installed and up-to-date.
- The accuracy and performance metrics displayed are based on pre-defined values and confusion matrices.

