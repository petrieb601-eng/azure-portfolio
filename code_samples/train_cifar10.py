import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ── Paths ──────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, "data")
model_path = os.path.join(script_dir, "cifar10_model.pth")

# ── Data ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=64,
                                          shuffle=False)

classes = ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck')

# ── Model ──────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 512)
        self.fc2   = nn.Linear(512, 10)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model     = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS = 10

print(f"Training for {EPOCHS} epochs on CPU...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  "
                  f"Step {i+1}/{len(trainloader)}  "
                  f"Loss: {running_loss/100:.3f}")
            running_loss = 0.0

print("Training complete.")

# ── Evaluation ─────────────────────────────────────────────────────────────
correct = 0
total   = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        outputs    = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total      += labels.size(0)
        correct    += (predicted == labels).sum().item()

print(f"Test accuracy: {100 * correct / total:.2f}%")

# ── Save ───────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")