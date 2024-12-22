import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#refer to:https://teddykoker.com/2019/06/trading-with-reinforcement-learning-in-python-part-ii-application/?ref=blog.mlq.ai

# Load and prepare data
def load_data(file_path = "RL\QQQ_twelve_data_filled_30_indicators.csv"):
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    # Keep only the required columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.index = pd.to_datetime(df.index, unit='s')
    return df

# Define the Sharpe ratio function
def sharpe_ratio(rets):
    return rets.mean() / rets.std()

# Define positions function
def positions(x, theta):
    M = len(theta) - 2
    T = len(x)
    Ft = np.zeros(T)
    for t in range(M, T):
        xt = np.concatenate([[1], x[t - M:t], [Ft[t - 1]]])
        Ft[t] = np.tanh(np.dot(theta, xt))
    return Ft

# Define returns function
def returns(Ft, x, delta):
    T = len(x)
    rets = Ft[0:T - 1] * x[1:T] - delta * np.abs(Ft[1:T] - Ft[0:T - 1])
    return np.concatenate([[0], rets])

# Define gradient function
def gradient(x, theta, delta):
    Ft = positions(x, theta)
    rets = returns(Ft, x, delta)
    T = len(x)
    M = len(theta) - 2

    A = np.mean(rets)
    B = np.mean(np.square(rets))
    S = A / np.sqrt(B - A ** 2)

    grad = np.zeros(M + 2)  # Initialize gradient
    dFpdtheta = np.zeros(M + 2)  # For storing previous dFdtheta

    for t in range(M, T):
        xt = np.concatenate([[1], x[t - M:t], [Ft[t - 1]]])
        dRdF = -delta * np.sign(Ft[t] - Ft[t - 1])
        dRdFp = x[t] + delta * np.sign(Ft[t] - Ft[t - 1])
        dFdtheta = (1 - Ft[t] ** 2) * (xt + theta[-1] * dFpdtheta)
        dSdtheta = (dRdF * dFdtheta + dRdFp * dFpdtheta)
        grad = grad + dSdtheta
        dFpdtheta = dFdtheta

    return grad, S

# Define training function
def train(x, epochs=500, M=5, commission=0.0025, learning_rate=0.1):
    theta = np.ones(M + 2)
    sharpes = np.zeros(epochs)  # Store Sharpe ratios over time
    for i in range(epochs):
        grad, sharpe = gradient(x, theta, commission)
        theta = theta + grad * learning_rate

        sharpes[i] = sharpe

    print("Finished training")
    return theta, sharpes

#  train the model on N samples, and then trade on the next P samples.
# thus split the data into training and test data, then normalize with training data:
df = load_data()
rets = df['close'].diff()
x = np.array(rets)

N = 4000 # set to 3500 on 3800d, 1d, 4000 on 5m, 60d
P = 200 # for now set to 200 on both daily and 5m
x_train = x[-(N+P):-P]
x_test = x[-P:]

std = np.std(x_train)
mean = np.mean(x_train)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

theta, sharpes = train(x_train, epochs=500, M=8, commission=0.0025, learning_rate=.001)

plt.plot(sharpes)
plt.xlabel('Epoch Number')
plt.ylabel('Sharpe Ratio')
plt.show()

test_returns = returns(positions(x_test, theta), x_test, 0.0025)
plt.plot((test_returns).cumsum(), label="Reinforcement Learning Model", linewidth=1)
plt.plot(x_test.cumsum(), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend()
plt.title("RL Model vs. Buy and Hold - Test Data");
plt.show()