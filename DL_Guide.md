# Deep Learning Practicals — Detailed Beginner Guide

---

## BASICS FIRST: What is Deep Learning?

Deep Learning = Teaching a computer to learn patterns from data using **Neural Networks**.

A Neural Network has:
- **Neurons** — small units that take numbers in, do math, give a number out
- **Layers** — groups of neurons. Input Layer → Hidden Layers → Output Layer
- **Weights & Bias** — numbers the model learns during training
- **Activation Function** — decides if a neuron should "fire" or not

**How training works (simple version):**
1. Give data to model → model makes a **prediction**
2. Compare prediction with actual answer → calculate **loss (error)**
3. Adjust weights to reduce error → this is called **backpropagation**
4. Repeat many times (epochs) until error is small

**Key terms you must know:**

| Term | Simple Meaning |
|------|---------------|
| **Epoch** | One complete pass through all training data |
| **Batch Size** | How many samples to process before updating weights |
| **Loss** | How wrong the model is (lower = better) |
| **Optimizer (Adam)** | Algorithm that adjusts weights smartly |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Train-Test Split** | Keep some data aside to test if model truly learned |

---

## Common Pattern in ALL 4 DL Practicals

Every practical follows the same 7 steps:
```
1. Import libraries
2. Load dataset
3. Preprocess data (clean, scale, reshape)
4. Split into train & test
5. Build model
6. Train model (fit)
7. Evaluate & plot results
```

---

# PRACTICAL 1: Boston House Price Prediction

## What are we doing?
Predicting the **price of a house** in Boston based on 13 features like crime rate, number of rooms, etc.

## What type of problem?
**Regression** — the output is a continuous number (price), not a category.

## What model?
A single neuron with linear activation = **Linear Regression** using neural network.

## Dataset
- 506 houses, each with 13 features
- Target column: MEDV (Median house value in $1000s)
- Important features: RM (rooms) → more rooms = higher price, LSTAT (poverty %) → more poverty = lower price

---

## Line-by-Line Code Explanation

### Step 1: Import Libraries
```python
import numpy as np
```
- `numpy` = library for math operations (arrays, matrices)

```python
import pandas as pd
```
- `pandas` = library for loading and working with data tables (like Excel)

```python
import matplotlib.pyplot as plt
```
- `matplotlib` = library for drawing graphs and charts

```python
from sklearn.model_selection import train_test_split
```
- `train_test_split` = function that splits data into training set and testing set

```python
from sklearn.preprocessing import StandardScaler
```
- `StandardScaler` = makes all features have same scale (mean=0, std=1)

```python
from tensorflow.keras.models import Sequential
```
- `Sequential` = a way to build neural network by adding layers one by one

```python
from tensorflow.keras.layers import Dense
```
- `Dense` = a fully connected layer (every input connects to every neuron)

---

### Step 2: Load Dataset
```python
df = pd.read_csv("1_boston_housing.csv")
```
- Reads the CSV file and stores it as a table called `df` (dataframe)

```python
df.columns = df.columns.str.replace('"', '')
```
- Removes extra quote marks from column names (cleaning step)

```python
df.head()
```
- Shows first 5 rows of the data to check if loading worked

---

### Step 3: Separate Features and Target
```python
X = df.drop("MEDV", axis=1)
```
- `X` = all columns EXCEPT MEDV (these are the 13 input features)
- `axis=1` means drop a column (axis=0 would drop a row)

```python
y = df["MEDV"]
```
- `y` = only the MEDV column (the price we want to predict)

```python
print(X.shape, y.shape)
```
- Shows (506, 13) and (506,) — 506 houses, 13 features each

---

### Step 4: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
- Splits data: 80% for training, 20% for testing
- `test_size=0.2` = 20% goes to test
- `random_state=42` = same split every time you run (for reproducibility)
- `X_train` = training features, `y_train` = training prices
- `X_test` = test features, `y_test` = test prices (model never sees these during training)

---

### Step 5: Feature Scaling
```python
scaler = StandardScaler()
```
- Creates a scaler object

```python
X_train = scaler.fit_transform(X_train)
```
- `fit` = learns the mean and standard deviation from training data
- `transform` = applies the formula: `(value - mean) / std` to each feature
- After this, all features have mean=0 and std=1

```python
X_test = scaler.transform(X_test)
```
- ONLY transform (no fit!) — uses the SAME mean/std learned from training data
- **WHY?** If we fit on test data too, we would be "cheating" — this is called data leakage

**Why scale at all?** Feature RM (rooms) ranges 3-9, but TAX ranges 180-700. Without scaling, the model would think TAX is more important just because its numbers are bigger.

---

### Step 6: Build the Model
```python
model = Sequential()
```
- Creates an empty neural network (we will add layers to it)

```python
model.add(Dense(1, input_shape=(X_train.shape[1],), activation='linear'))
```
- `Dense(1)` = adds ONE neuron
- `input_shape=(13,)` = this neuron takes 13 inputs (one per feature)
- `activation='linear'` = output = weighted sum as-is (no transformation)
- This single neuron computes: `price = w1*CRIM + w2*ZN + ... + w13*LSTAT + bias`
- Total parameters: 13 weights + 1 bias = **14**

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```
- `optimizer='adam'` = how to adjust weights (Adam = smart gradient descent)
- `loss='mse'` = Mean Squared Error = average of (actual - predicted)²
- `metrics=['mae']` = also track Mean Absolute Error = average of |actual - predicted|

```python
model.summary()
```
- Prints the model structure and total trainable parameters (14)

---

### Step 7: Train the Model
```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)
```
- `X_train, y_train` = data to train on
- `epochs=100` = go through all training data 100 times
- `batch_size=16` = process 16 houses, then update weights, repeat
- `validation_split=0.2` = keep 20% of training data to check for overfitting
- `verbose=1` = show progress bar during training
- `history` = stores loss values at each epoch (for plotting later)

---

### Step 8: Evaluate
```python
loss, mae = model.evaluate(X_test, y_test)
```
- Tests the model on data it has NEVER seen
- Returns MSE (loss) and MAE

```python
print("Test Loss (MSE):", loss)
print("Test MAE:", mae)
```
- MAE of 3.5 means on average, prediction is off by $3,500

---

### Step 9: Predict
```python
y_pred = model.predict(X_test)
```
- Makes predictions for all test houses

```python
for i in range(5):
    print(f"Actual: {y_test.iloc[i]:.2f} | Predicted: {y_pred[i][0]:.2f}")
```
- Shows first 5 actual vs predicted prices to compare

---

### Step 10: Plot
```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
```
- Plots how loss decreased over 100 epochs
- Both lines should go DOWN → model is learning
- If training loss goes down but validation goes UP → overfitting

---
---

# PRACTICAL 2: IMDB Sentiment Classification

## What are we doing?
Reading movie reviews and predicting if they are **Positive** or **Negative**.

## What type of problem?
**Binary Classification** — two classes: Positive (1) and Negative (0).

## What model?
A Deep Neural Network (DNN) with 2 hidden layers.

## Dataset
- 50,000 movie reviews with labels ('positive' or 'negative')
- Text data — needs to be converted to numbers first

---

## Line-by-Line Code Explanation

### Step 1: Imports
```python
import re
```
- `re` = Regular Expressions library — used for text cleaning (finding and removing patterns)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
- `TfidfVectorizer` = converts text into numbers using TF-IDF method

**What is TF-IDF?**
- TF (Term Frequency) = how many times a word appears in one review
- IDF (Inverse Document Frequency) = how rare the word is across all reviews
- TF × IDF = importance score. Common words like "the" get low score. Special words like "masterpiece" get high score.

---

### Step 2: Load Dataset
```python
df = pd.read_csv(
    "IMDB_Dataset.csv",
    encoding_errors='ignore',
    on_bad_lines='skip',
    engine='python'
)
```
- `encoding_errors='ignore'` = skip characters that can't be read
- `on_bad_lines='skip'` = skip broken rows instead of crashing
- `engine='python'` = use Python's CSV reader (more flexible)

---

### Step 3: Fix Column Names
```python
df.columns = ['review', 'sentiment']
```
- Renames columns to simple names we can use

---

### Step 4: Clean Text
```python
def clean_text(text):
    text = str(text)
```
- Converts to string (in case of NaN values)

```python
    text = text.lower()
```
- Makes everything lowercase: "GREAT Movie" → "great movie"

```python
    text = re.sub(r'<.*?>', '', text)
```
- Removes HTML tags: `"<br>hello<br>"` → `"hello"`
- `<.*?>` matches anything between < and >

```python
    text = re.sub(r'[^a-zA-Z ]', '', text)
```
- Removes everything that is NOT a letter or space
- "good!!! movie..." → "good movie"

```python
df['review'] = df['review'].apply(clean_text)
```
- Applies the cleaning function to every single review

---

### Step 5: Convert Labels
```python
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
```
- Neural networks need numbers, not words
- 'positive' → 1, 'negative' → 0

---

### Step 6: TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(max_features=5000)
```
- Creates a TF-IDF converter that keeps only top 5000 words

```python
X = vectorizer.fit_transform(df['review']).toarray()
```
- Converts each review into a vector of 5000 numbers
- Each number = importance of that word in that review
- `.toarray()` = converts sparse matrix to regular array

```python
y = df['sentiment']
```
- Labels: 0 or 1

---

### Step 7: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- 80% train, 20% test (same as Boston)

---

### Step 8: Build DNN Model
```python
model = Sequential()
```

```python
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
```
- First hidden layer: 128 neurons
- `input_shape=(5000,)` = takes 5000 TF-IDF features as input
- `relu` = ReLU activation: if value > 0, keep it. If < 0, make it 0.
- Parameters: 5000 × 128 + 128 = 640,128

```python
model.add(Dense(64, activation='relu'))
```
- Second hidden layer: 64 neurons
- Parameters: 128 × 64 + 64 = 8,256

```python
model.add(Dense(1, activation='sigmoid'))
```
- Output layer: 1 neuron with sigmoid activation
- Sigmoid squeezes output between 0 and 1 (probability)
- If output > 0.5 → Positive. If < 0.5 → Negative.
- Parameters: 64 × 1 + 1 = 65

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```
- `binary_crossentropy` = loss function for 2-class problems
- `accuracy` = percentage of correct predictions

---

### Step 9: Train
```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```
- 10 epochs, batch size 32
- `validation_data` = use test data to monitor performance each epoch

---

### Step 10: Plot & Evaluate
```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```
- Shows how accuracy improves over epochs

```python
loss, accuracy = model.evaluate(X_test, y_test)
print("Final Accuracy:", accuracy)
```
- Typically 85-88% accuracy

---
---

# PRACTICAL 3: Fashion MNIST CNN

## What are we doing?
Looking at 28×28 pixel images of clothing and classifying them into 10 categories.

## What type of problem?
**Multi-class Classification** — 10 classes (T-shirt, Trouser, Dress, etc.)

## What model?
**CNN (Convolutional Neural Network)** — designed specifically for images.

## Why CNN instead of DNN?
DNN treats each pixel independently — it doesn't know that neighboring pixels form shapes. CNN slides small filters across the image to detect edges, shapes, and patterns.

## Dataset
- 60,000 training images + 10,000 test images
- Each image: 28 × 28 pixels, grayscale (black & white)
- 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## New Layers You Need to Know

| Layer | What it does (simple) |
|-------|----------------------|
| **Conv2D** | Slides a small filter (3×3) over the image to find features like edges |
| **MaxPooling2D** | Shrinks the image by keeping only the biggest value in each small block |
| **Dropout** | Randomly switches off some neurons during training to prevent overfitting |
| **Flatten** | Converts 2D image data into a 1D list so Dense layer can process it |

---

## Line-by-Line Code Explanation

### Step 1: Imports
```python
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)
```
- Importing all CNN-specific layers along with Dense

---

### Step 2: Load Data
```python
train_df = pd.read_csv('fashion_train.csv')
test_df  = pd.read_csv('fashion_test.csv')
```
- `train_df.shape` = (60000, 785) → 60000 images, 785 columns
- Column 0 = label (0-9), Columns 1-784 = pixel values (28×28 = 784 pixels)

```python
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```
- Human-readable names for labels 0-9

---

### Step 3: Prepare Data
```python
X_train = train_df.iloc[:, 1:].to_numpy()
```
- Takes all columns EXCEPT first (all pixel values)
- `.to_numpy()` converts pandas dataframe to numpy array

```python
y_train = train_df.iloc[:, 0].to_numpy()
```
- Takes only first column (labels)

```python
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
```
- `reshape(-1, 28, 28, 1)`:
  - `-1` = automatically calculate (60000)
  - `28, 28` = image height and width
  - `1` = 1 color channel (grayscale). RGB would be 3.
- `/ 255.0` = pixels go from 0-255 to 0-1 (normalization)

**Why reshape?** The CSV has each image as a flat row of 784 numbers. Conv2D needs a 2D image format: (height, width, channels).

---

### Step 4: Visualize Samples
```python
plt.figure(figsize=(8, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(class_names[y_train[i]], fontsize=8)
    plt.axis('off')
plt.show()
```
- Shows 5 sample images with their labels to verify data loaded correctly

---

### Step 5: Build CNN Model
```python
model = Sequential([
```

```python
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
```
- `64` = number of filters (64 different feature detectors)
- `(3, 3)` = each filter is 3×3 pixels
- Each filter slides across the 28×28 image and produces a feature map
- Output size: 28 - 3 + 1 = **26 × 26 × 64**

```python
    MaxPooling2D((2, 2)),
```
- Takes a 2×2 window, keeps only the MAX value
- Reduces 26×26 → **13×13** (halves each dimension)
- Keeps the strongest features, throws away less important details

```python
    Dropout(0.3),
```
- During training, randomly turns OFF 30% of neurons
- Forces the network to not rely on any single neuron → prevents overfitting
- During testing/prediction, Dropout is automatically TURNED OFF

```python
    Flatten(),
```
- Converts 13 × 13 × 64 = 10,816 values into a single flat list
- Needed because Dense layers expect 1D input

```python
    Dense(32, activation='relu'),
```
- 32 neurons, learns combinations of features from Conv layers

```python
    Dense(10, activation='softmax')
```
- 10 neurons (one per class)
- **Softmax** = converts raw scores into probabilities that sum to 1.0
- Example output: [0.01, 0.02, 0.8, 0.01, 0.05, 0.01, 0.03, 0.02, 0.04, 0.01]
- Highest = index 2 = Pullover

```python
])
```

---

### Step 6: Compile
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
- `sparse_categorical_crossentropy`:
  - "sparse" = labels are integers (0, 1, 2...9)
  - "categorical" = multiple categories
  - "crossentropy" = measures how far predicted probabilities are from actual class

---

### Step 7: Train
```python
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
```
- Only 5 epochs needed — CNN learns fast on this dataset
- Achieves ~88-91% accuracy

---

### Step 8: Evaluate & Predict
```python
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
```

```python
predictions = model.predict(X_test).argmax(axis=1)
```
- `model.predict` gives 10 probabilities per image
- `.argmax(axis=1)` picks the index with highest probability = predicted class

---
---

# PRACTICAL 4: Google Stock Price Prediction (RNN)

## What are we doing?
Using past 60 days of stock prices to **predict the next day's price**.

## What type of problem?
**Time-Series Regression** — predicting a future continuous value from past sequence.

## What model?
**RNN (Recurrent Neural Network)** — has memory to remember past data points.

## Why RNN?
Stock prices are sequential — today's price depends on previous days. DNN treats each input independently and has no concept of "order" or "time". RNN processes data step-by-step and carries a **hidden state** (memory) forward.

---

## Line-by-Line Code Explanation

### Step 1: Imports
```python
from sklearn.preprocessing import MinMaxScaler
```
- `MinMaxScaler` = scales data to range [0, 1]

```python
from tensorflow.keras.layers import Dense, SimpleRNN
```
- `SimpleRNN` = basic RNN layer with memory

---

### Step 2: Load Dataset
```python
df = pd.read_csv("Google_Stock_Price.csv", thousands=',')
```
- `thousands=','` = handles numbers like "1,234.56" by removing commas

```python
data = pd.to_numeric(df['Open'], errors='coerce').dropna().values.reshape(-1, 1)
```
- `pd.to_numeric` = converts the 'Open' price column to numbers
- `errors='coerce'` = if a value can't be converted (like text), make it NaN
- `.dropna()` = remove NaN values
- `.reshape(-1, 1)` = make it a column vector (needed for scaler)

---

### Step 3: Scale Data
```python
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
```
- Scales all prices to range [0, 1]
- Example: if min price=50, max=1500, then price 775 becomes 0.5
- **Why?** RNN works much better with small values in [0,1]

---

### Step 4: Train-Test Split
```python
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]
```
- First 80% = training, Last 20% = testing
- **NO random shuffle!** Time-series must stay in order

---

### Step 5: Create Sliding Window Dataset
```python
def create_dataset(dataset):
    X = []
    y = []
    for i in range(60, len(dataset)):
        X.append(dataset[i-60:i, 0])   # past 60 days = input
        y.append(dataset[i, 0])         # day 61 = target
    return np.array(X), np.array(y)
```
- This is the **key concept** of this practical
- For each day (starting from day 61), we look back 60 days
- Those 60 prices = input (X), the next price = target (y)

```
Example:
X[0] = [price_day1, price_day2, ..., price_day60]  → y[0] = price_day61
X[1] = [price_day2, price_day3, ..., price_day61]  → y[1] = price_day62
X[2] = [price_day3, price_day4, ..., price_day62]  → y[2] = price_day63
```

```python
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)
```

---

### Step 6: Reshape for RNN
```python
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```
- RNN needs 3D input: `(number_of_samples, timesteps, features_per_step)`
- `X_train.shape[0]` = number of samples
- `X_train.shape[1]` = 60 (timesteps)
- `1` = we have 1 feature per timestep (just the price)

---

### Step 7: Build RNN Model
```python
model = Sequential()
```

```python
model.add(SimpleRNN(50, return_sequences=True, input_shape=(60, 1)))
```
- `SimpleRNN(50)` = RNN layer with 50 hidden units (memory size)
- `input_shape=(60, 1)` = 60 timesteps, 1 feature each
- `return_sequences=True` = output the hidden state at EVERY timestep
- **Why True?** Because another RNN layer comes next — it needs input at every step
- Output shape: (60, 50) — 50 values at each of 60 timesteps

```python
model.add(SimpleRNN(50))
```
- Second RNN layer, also 50 units
- `return_sequences=False` (default) = output ONLY the last hidden state
- **Why False?** Because next is a Dense layer — it only needs one vector
- Output shape: (50,) — single vector of 50 values

```python
model.add(Dense(1))
```
- One neuron, linear activation (default) → predicts one price value

```python
model.compile(optimizer='adam', loss='mean_squared_error')
```
- MSE loss because this is regression (predicting a number)

---

### Step 8: Train
```python
model.fit(X_train, y_train, epochs=20, batch_size=32)
```
- 20 epochs, batch size 32

---

### Step 9: Predict & Convert Back
```python
predicted = model.predict(X_test)
```
- Predictions are in scaled [0,1] range

```python
predicted = scaler.inverse_transform(predicted)
```
- Converts [0,1] back to actual dollar prices
- Example: 0.5 → $775 (if original range was $50-$1500)

```python
real = scaler.inverse_transform(y_test.reshape(-1, 1))
```
- Same for actual prices

---

### Step 10: Plot
```python
plt.plot(real, color='red', label='Real Price')
plt.plot(predicted, color='blue', label='Predicted Price')
plt.title("Google Stock Price Prediction (RNN)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
```
- Red line = actual prices, Blue line = predicted prices
- If blue follows red closely → model is good
