import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

data_path = "./data/regression"
dataset_name = sys.argv[1]

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load CSV
data = pd.read_csv(f"{data_path}/{dataset_name}")
seed = 2137

torch.manual_seed(seed)

# 2. Split features and target
X = data.drop("SalePrice", axis=1).values
y = data["SalePrice"].values

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# 4. Feature scaling (important for regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# 6. Define model
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


model = RegressionModel(X_train.shape[1])

# 7. Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# move stuff to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
model = model.to(device)

# 8. Training loop
epochs = 10000

for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 9. Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)

    # Convert tensors to NumPy
    y_pred = torch.Tensor.cpu(predictions)
    y_pred = y_pred.numpy().flatten()
    y_test = torch.Tensor.cpu(y_test)
    y_true = y_test.numpy().flatten()

    # Mean Absolute Error easier to interpret in units of SalePrice
    mae = mean_absolute_error(y_true, y_pred)

    # Root Mean Squared Error similar to MSE but in original scale
    rmse = mean_squared_error(y_true, y_pred)

    # R^2 Score – proportion of variance explained
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 Score: {r2:.3f}")

print(f"\nTest MSE: {test_loss.item():.4f}")
