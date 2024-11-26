from src.data.load import DataLoader
from src.data.prepare import Prepare
from src.agents.dql.train import Train as DeepQL
from src.agents.mlp_encoder.train import Train as SimpleMLP
from src.agents.mlp_windowed_custom.train import Train as WindowedMLP

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random
import time
from datetime import timedelta

start_time = time.time()

# Core Training Parameters
BATCH_SIZE = 128        # Number of samples per training iteration
                      # Ideal: 32-128. Larger = more stable but slower
                      # GPU can handle larger batches (64-256)

GAMMA = 0.95           # Discount factor for future rewards (0-1)
                      # Ideal: 0.95-0.99 for trading
                      # Higher (0.99) for long-term strategies
                      # Lower (0.8-0.9) for short-term trading

n_step = 4            # Steps ahead for calculating returns
                      # Ideal: 3-5 for trading
                      # Higher (5-10) for trending markets
                      # Lower (1-3) for volatile markets

n_episodes = 200       # Number of training iterations
                      # Ideal: 200-1000 for complex markets
                      # Higher (500+) for more robust strategies
                      # Current 50 is quite low for production

# Memory/Update Parameters
ReplayMemorySize = 10000 # Size of experience replay buffer
                      # Ideal: 10000-100000
                      # Current 36 is very low
                      # Larger (50000+) for more stable learning

TARGET_UPDATE = 15     # How often to update target network
                      # Ideal: 10-20 episodes
                      # Higher (15-20) for more stability
                      # Lower (5-10) for faster adaptation

# Market Parameters
transaction_cost = 0.001 # Trading fees per transaction
                      # Ideal: 0.001-0.003 (0.1-0.3%) for stocks
                      # 0.0 is unrealistic - real markets have fees

initial_investment = 1000 # Starting capital for backtesting
                         # Ideal: Depends on strategy
                         # Common: 10000-100000 for realistic testing

# Data Processing Parameters
window_size = 30      # Past timesteps to consider
                      # Ideal: 20-50 for daily data
                      # Higher (40-60) for longer trends
                      # Lower (10-20) for quick responses

trend_size = 72       # Size of trend patterns to detect
                      # Ideal: 50-100 for daily data
                      # Higher (100+) for longer patterns
                      # Lower (30-50) for shorter patterns

indicators = True     # Include technical indicators
                      # Ideal: True for most strategies
                      # False only if using pure price action

# Data params
ticker = 'SPY'
begin_date = '2015-11-25'
end_date = '2024-11-25'
total_days = pd.to_datetime(end_date) - pd.to_datetime(begin_date)
split_point = pd.to_datetime(begin_date) + (total_days * 0.7)
split_point = split_point.strftime('%Y-%m-%d')

train_portfolios = {}
test_portfolios = {}

# Create local output directory
output_dir = 'Results'
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess data
loader = DataLoader(
    dataset_name=ticker,
    split_point=split_point,
    begin_date=begin_date,
    end_date=end_date,
    load_from_file=False,
    crypto=False,
    include_indicators=indicators
)


def add_train_portfo(model_name, portfo):
    counter = 0
    key = f'{model_name}'
    while key in train_portfolios.keys():
        counter += 1
        key = f'{model_name}{counter}'
    train_portfolios[key] = portfo


def add_test_portfo(model_name, portfo):
    counter = 0
    key = f'{model_name}'
    while key in test_portfolios.keys():
        counter += 1
        key = f'{model_name}{counter}'
    test_portfolios[key] = portfo


data_trainSimp = loader.data_train
data_testSimp = loader.data_test


# Data for autopattern agents
dataTrain_autoPatternExtractionAgent = Prepare(
    data_trainSimp,
    'action',
    GAMMA,
    n_step,
    BATCH_SIZE,
    trend_size,
    transaction_cost,
    windowed=False
)

dataTest_autoPatternExtractionAgent = Prepare(
    data_testSimp,
    'action',
    GAMMA,
    n_step,
    BATCH_SIZE,
    trend_size,
    transaction_cost,
    windowed=False
)

# 1. DQN-ohlc+indicators
deepRLAgent = DeepQL(
    loader,
    dataTrain_autoPatternExtractionAgent,
    dataTest_autoPatternExtractionAgent,
    ticker,
    window_size,
    transaction_cost,
    BATCH_SIZE=BATCH_SIZE,
    GAMMA=GAMMA,
    ReplayMemorySize=ReplayMemorySize,
    TARGET_UPDATE=TARGET_UPDATE,
    n_step=n_step
)

deepRLAgent.train(n_episodes)
model_type = 'DQN-ohlc+indicators'

ev_deepRLAgentAutoExtraction_train = deepRLAgent.test(
    initial_investment=initial_investment,
    test_type='train'
)
ev_deepRLAgentAutoExtraction_test = deepRLAgent.test(
    initial_investment=initial_investment,
    test_type='test'
)

add_train_portfo(
    model_type, ev_deepRLAgentAutoExtraction_train.get_daily_portfolio_value())
add_test_portfo(
    model_type, ev_deepRLAgentAutoExtraction_test.get_daily_portfolio_value())

# 2. MLP-ohlc+indicators
n_classes = 64

simpleMLPAgent = SimpleMLP(
    loader,
    dataTrain_autoPatternExtractionAgent,
    dataTest_autoPatternExtractionAgent,
    ticker,
    window_size,
    transaction_cost,
    n_classes,
    BATCH_SIZE=BATCH_SIZE,
    GAMMA=GAMMA,
    ReplayMemorySize=ReplayMemorySize,
    TARGET_UPDATE=TARGET_UPDATE,
    n_step=n_step
)

simpleMLPAgent.train(n_episodes)
model_type = 'MLP-ohlc+indicators'

ev_simpleMLP_train = simpleMLPAgent.test(
    initial_investment=initial_investment,
    test_type='train'
)

ev_simpleMLP_test = simpleMLPAgent.test(
    initial_investment=initial_investment,
    test_type='test'
)

add_train_portfo(model_type, ev_simpleMLP_train.get_daily_portfolio_value())
add_test_portfo(model_type, ev_simpleMLP_test.get_daily_portfolio_value())


# custom windowed
loader = DataLoader(
    dataset_name=ticker,
    split_point=split_point,
    begin_date=begin_date,
    end_date=end_date,
    load_from_file=False,
    crypto=False,
    include_indicators=indicators
)


data_trainWind = loader.data_train
data_testWind = loader.data_test

# 3. MLP-ohlc+custom window
dataTrain_windowedAgent = Prepare(
    data_trainWind,
    'action',
    GAMMA,
    n_step,
    BATCH_SIZE,
    trend_size,
    transaction_cost,
    windowed=True
)

dataTest_windowedAgent = Prepare(
    data_testWind,
    'action',
    GAMMA,
    n_step,
    BATCH_SIZE,
    trend_size,
    transaction_cost,
    windowed=True
)

windowedMLPAgent = WindowedMLP(
    loader,
    dataTrain_windowedAgent,
    dataTest_windowedAgent,
    ticker,
    window_size,
    transaction_cost,
    n_classes,
    BATCH_SIZE=BATCH_SIZE,
    GAMMA=GAMMA,
    ReplayMemorySize=ReplayMemorySize,
    TARGET_UPDATE=TARGET_UPDATE,
    n_step=n_step
)

windowedMLPAgent.train(n_episodes)
model_type = 'MLP-windowed-custom'

ev_windowedMLP_train = windowedMLPAgent.test(
    initial_investment=initial_investment,
    test_type='train'
)

ev_windowedMLP_test = windowedMLPAgent.test(
    initial_investment=initial_investment,
    test_type='test'
)

add_train_portfo(model_type, ev_windowedMLP_train.get_daily_portfolio_value())
add_test_portfo(model_type, ev_windowedMLP_test.get_daily_portfolio_value())

# Plotting for train data
experiment_num = 1
RESULTS_PATH = 'TestResults/Train/'

os.makedirs(RESULTS_PATH, exist_ok=True)

while os.path.exists(f'{RESULTS_PATH}{ticker};train;EXPERIMENT({experiment_num}).jpg'):
    experiment_num += 1

fig_file = f'{RESULTS_PATH}{ticker};train;EXPERIMENT({experiment_num}).jpg'

sns.set(rc={'figure.figsize': (15, 7)})

items = list(train_portfolios.keys())
random.shuffle(items)

first = True
for k in items:
    profit_percentage = [(train_portfolios[k][i] - train_portfolios[k][0])/train_portfolios[k][0] * 100
                         for i in range(len(train_portfolios[k]))]
    difference = len(train_portfolios[k]) - len(loader.data_train_with_date)
    df = pd.DataFrame({'date': loader.data_train_with_date.index,
                       'portfolio': profit_percentage[difference:]})
    if not first:
        df.plot(ax=ax, x='date', y='portfolio', label=k)
    else:
        ax = df.plot(x='date', y='portfolio', label=k)
        first = False

ax.set(xlabel='Time', ylabel='%Rate of Return')
ax.set_title(
    f'%Rate of Return at each point of time for training data of {ticker}')

plt.legend()
plt.savefig(fig_file, dpi=300)
plt.close()

# Plotting for test data
experiment_num = 1
RESULTS_PATH = 'TestResults/Test/'

os.makedirs(RESULTS_PATH, exist_ok=True)

while os.path.exists(f'{RESULTS_PATH}{ticker};test;EXPERIMENT({experiment_num}).jpg'):
    experiment_num += 1

fig_file = f'{RESULTS_PATH}{ticker};test;EXPERIMENT({experiment_num}).jpg'

sns.set(rc={'figure.figsize': (15, 7)})
sns.set_palette(sns.color_palette("Paired", 15))

items = list(test_portfolios.keys())
random.shuffle(items)

first = True
for k in items:
    profit_percentage = [(test_portfolios[k][i] - test_portfolios[k][0])/test_portfolios[k][0] * 100
                         for i in range(len(test_portfolios[k]))]
    difference = len(test_portfolios[k]) - len(loader.data_test_with_date)
    df = pd.DataFrame({'date': loader.data_test_with_date.index,
                       'portfolio': profit_percentage[difference:]})
    if not first:
        df.plot(ax=ax, x='date', y='portfolio', label=k)
    else:
        ax = df.plot(x='date', y='portfolio', label=k)
        first = False

ax.set(xlabel='Time', ylabel='%Rate of Return')
ax.set_title(f'Comparing the %Rate of Return for different models '
             f'at each point of time for test data of {ticker}')
plt.legend()
plt.savefig(fig_file, dpi=300)
plt.close()

print(
    f"Train plot saved as {RESULTS_PATH}{ticker};train;EXPERIMENT({experiment_num}).jpg")
print(
    f"Test plot saved as {RESULTS_PATH}{ticker};test;EXPERIMENT({experiment_num}).jpg")

total_time = time.time() - start_time
print(f"\nTotal training time: {str(timedelta(seconds=total_time))}")