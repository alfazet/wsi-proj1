import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Load data and set seed
# ------------------------------
data = pd.read_csv("data/classification/dataset.csv")
seed = 2137
torch.manual_seed(seed)

# ------------------------------
# 2. Split features and target
# ------------------------------
X = data.drop("growth direction", axis=1).values
y = data["growth direction"].values

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split (stratify to preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
)

# ------------------------------
# 3. Scale features
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# 4. Convert to PyTorch tensors
# ------------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# ------------------------------
# 5. Define model
# ------------------------------
class ClassificationModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.model(x)


num_classes = len(le.classes_)
model = ClassificationModel(X_train.shape[1], num_classes)

# ------------------------------
# 6. Loss and optimizer
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# 7. Training loop
# ------------------------------
epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# ------------------------------
# 8. Evaluation
# ------------------------------
model.eval()
with torch.no_grad():
    logits = model(X_test)
    predictions = torch.argmax(logits, dim=1)

accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=le.classes_))
