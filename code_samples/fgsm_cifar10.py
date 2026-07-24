import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, "data")
model_path = os.path.join(script_dir, "cifar10_model.pth")

# ── Model ──────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1   = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(64 * 8 * 8, 512)
        self.fc2     = nn.Linear(512, 10)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ── Load model ─────────────────────────────────────────────────────────────
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
print("Model loaded.")

# ── Data ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset    = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                          download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck')

criterion = nn.CrossEntropyLoss()

# ── FGSM function ──────────────────────────────────────────────────────────
def fgsm_attack(image, epsilon, gradient):
    # Take the sign of the gradient and scale by epsilon
    perturbation    = epsilon * gradient.sign()
    # Add perturbation to image
    adversarial_image = image + perturbation
    # Clip to keep pixel values in valid range [-1, 1] after normalization
    adversarial_image = torch.clamp(adversarial_image, -1, 1)
    return adversarial_image

# ── Attack loop ────────────────────────────────────────────────────────────
def run_attack(epsilon):
    correct_clean = 0
    correct_adv   = 0
    total         = 0

    for images, labels in testloader:
        # ── Clean accuracy ─────────────────────────────────────────────────
        images.requires_grad = True
        outputs = model(images)
        _, predicted_clean = torch.max(outputs, 1)
        correct_clean += (predicted_clean == labels).sum().item()

        # ── Compute gradient ───────────────────────────────────────────────
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        gradient = images.grad.data

        # ── Generate adversarial images ────────────────────────────────────
        adv_images = fgsm_attack(images, epsilon, gradient)

        # ── Adversarial accuracy ───────────────────────────────────────────
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, predicted_adv = torch.max(adv_outputs, 1)
            correct_adv += (predicted_adv == labels).sum().item()

        total += labels.size(0)

    clean_acc = 100 * correct_clean / total
    adv_acc   = 100 * correct_adv   / total
    return clean_acc, adv_acc

# ── Test multiple epsilon values ───────────────────────────────────────────
epsilons = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]

print(f"\n{'Epsilon':<12} {'Clean Acc':<15} {'Adv Acc':<15} {'Drop'}")
print("-" * 50)

clean_accs = []
adv_accs   = []

for eps in epsilons:
    clean_acc, adv_acc = run_attack(eps)
    clean_accs.append(clean_acc)
    adv_accs.append(adv_acc)
    print(f"{eps:<12.3f} {clean_acc:<15.2f} {adv_acc:<15.2f} {clean_acc - adv_acc:.2f}%")

# ── Plot results ───────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(epsilons, clean_accs, "b-o", label="Clean accuracy")
plt.plot(epsilons, adv_accs,   "r-o", label="Adversarial accuracy")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy (%)")
plt.title("FGSM Attack on CIFAR-10")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(script_dir, "fgsm_cifar10_results.png"))
plt.show()
print("\nPlot saved to fgsm_cifar10_results.png")

# ── Visualize example adversarial images ──────────────────────────────────
def visualize_examples(epsilon, n=8):
    images, labels = next(iter(testloader))
    images.requires_grad = True

    outputs = model(images)
    loss    = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    gradient   = images.grad.data
    adv_images = fgsm_attack(images, epsilon, gradient)

    # Denormalize for display
    def denorm(tensor):
        return tensor * 0.5 + 0.5

    fig, axes = plt.subplots(2, n, figsize=(16, 4))
    for i in range(n):
        # Original
        orig = denorm(images[i]).detach().permute(1, 2, 0).numpy()
        orig = np.clip(orig, 0, 1)
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"True:\n{classes[labels[i]]}", fontsize=8)
        axes[0, i].axis("off")

        # Adversarial
        with torch.no_grad():
            adv_out  = model(adv_images[i].unsqueeze(0))
            pred_adv = torch.max(adv_out, 1)[1].item()
        adv = denorm(adv_images[i]).detach().permute(1, 2, 0).numpy()
        adv = np.clip(adv, 0, 1)
        axes[1, i].imshow(adv)
        axes[1, i].set_title(f"Pred:\n{classes[pred_adv]}", fontsize=8)
        axes[1, i].axis("off")

    plt.suptitle(f"Top: original  |  Bottom: adversarial (ε={epsilon})")
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f"fgsm_examples_eps{epsilon}.png"))
    plt.show()
    print(f"Example images saved.")

# Show examples at epsilon = 0.08
visualize_examples(0.08)