# Issues and Observations in Current Models

## LSTM Model Issues

### 1. LSTM with 30 Indicators vs. LSTM_8_Indicator
- **Observation**: The LSTM model with 30 indicators shows better performance compared to the LSTM_8_indicator model.
- **Key Differences**:
  - The LSTM_8_indicator model does not utilize cross-validation or GPU acceleration.

### 2. Unbalanced Cases
- **Issue**: The LSTM model does not explicitly address the problem of unbalanced classes in the dataset. This could lead to biased predictions.

### 3. Potential Overfitting
- **Observation**: While the performance of the LSTM model appears good, it may suffer from overfitting due to the model's complexity or the size of the dataset.

---

## CNN Model Issues

### 1. Data Preprocessing
- **Dataset Used**: `(data/QQQ_twelve_data_filled_30_indicators.csv)`.
- **Issue**: The original dataset contains missing values for some indicators. Missing values were filled using the `pandas_ta` library (refer to `fill_missing_values.py`).
- **Concern**:
  - Lack of domain knowledge in finance raises doubts about the correctness of the filled values.
  - Presence of extremely large values in the filled data, which might be problematic for the model.

### 2. Poor Model Performance
- **Observation**: The CNN model fails to learn effectively. The loss remains constant at 0.7, and predictions consistently indicate that the stock price will go up.
- **Possible Causes**:
  1. **Incorrect Data**: Issues mentioned in the data preprocessing step might have introduced noise.
  2. **Irrelevant Indicators**: Including all indicators without selection might ignore meaningful signals. Applying PCA or another feature selection method could help.
  3. **Ineffective Architecture**: The CNN architectures may not be effective. I tried several different CNN architectures but none of them has good performance.The current ResNet-like architecture also does not yield good results.
  4. **Normalization Issues**: Data normalization before training might reduce interpretability and fail to handle extreme values effectively.

### 3. Potential Fixes
- **Data**:
  - Explore the idea of adding more indicators.
  - Perform feature selection to identify the most relevant indicators before constructing the CNN model.
  - Refer to the approach described in [this blog post](https://medium.com/@quantclubiitkgpstock-buy-sell-hold-prediction-using-cnn-ee7b671f4ad3).
- **Architecture**:
  - Experiment with different CNN architectures (e.g., U-Net).
  - Fine-tune a pretrained model like MobileNet or PNASNet. However, resizing the input data to fit these models might be challenging.
- **Normalization**:
  - Revisit the normalization strategy to handle extreme values without compromising interpretability.
- **CNN**:
    - Evaluate whether CNN is suitable for this task, given that the input is a \(5 \times 6\) grayscale image. Compared to typical CNN applications like ImageNet and MNIST, the input size is too small. Consider a generalized linear model with odds ratio-based probability generation or other statistical methods might be more effective.

---

## Reinforcement Learning (RL) Issues

### 1. Implementation
- **Approach**: The RL implementation is inspired by [this tutorial](https://teddykoker.com/2019/06/trading-with-reinforcement-learning-in-python-part-ii-application/?ref=blog.mlq.ai).
- **Observation**: The performance  appears satisfactory, though further evaluation is required.

---

## General Issues

- **Lack of Finance Background**:
  - The absence of a finance background limits the ability to approach the problem. The project is treated purely as an engineering challenge, which makes me hard solve issues mantioned above.

---
