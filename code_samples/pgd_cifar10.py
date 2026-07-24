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

# ── PGD function ───────────────────────────────────────────────────────────
def pgd_attack(model, images, labels, epsilon, alpha, num_steps):
    """
    epsilon   = max total perturbation budget (ε ball)
    alpha     = step size per iteration
    num_steps = number of iterations
    """

    # Step 1 — random start inside the epsilon ball
    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, -1, 1) - images

    for step in range(num_steps):
        # Step 2 — enable gradient tracking on the perturbed image
        delta.requires_grad = True
        adv_images = images + delta

        # Step 3 — forward pass
        outputs = model(adv_images)
        loss    = criterion(outputs, labels)

        # Step 4 — backward pass to get gradient
        model.zero_grad()
        loss.backward()
        gradient = delta.grad.data

        # Step 5 — take one small FGSM step
        delta = delta.detach() + alpha * gradient.sign()

        # Step 6 — project back into epsilon ball (clip to stay within budget)
        delta = torch.clamp(delta, -epsilon, epsilon)

        # Step 7 — clip to keep valid pixel range
        delta = torch.clamp(images + delta, -1, 1) - images

    return images + delta.detach()

# ── Attack loop ────────────────────────────────────────────────────────────
def run_attack(epsilon, alpha, num_steps):
    correct_clean = 0
    correct_adv   = 0
    total         = 0

    for images, labels in testloader:

        # ── Clean accuracy ─────────────────────────────────────────────────
        with torch.no_grad():
            outputs = model(images)
            _, predicted_clean = torch.max(outputs, 1)
            correct_clean += (predicted_clean == labels).sum().item()

        # ── Generate adversarial images with PGD ───────────────────────────
        adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps)

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
# Alpha is typically epsilon/4 and steps is 40 for a strong attack
epsilons  = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
num_steps = 20

print(f"\nPGD Attack — {num_steps} steps, alpha = epsilon/4")
print(f"\n{'Epsilon':<12} {'Clean Acc':<15} {'Adv Acc':<15} {'Drop'}")
print("-" * 50)

clean_accs = []
adv_accs   = []

for eps in epsilons:
    alpha = eps / 4 if eps > 0 else 0.01
    clean_acc, adv_acc = run_attack(eps, alpha, num_steps)
    clean_accs.append(clean_acc)
    adv_accs.append(adv_acc)
    print(f"{eps:<12.3f} {clean_acc:<15.2f} {adv_acc:<15.2f} {clean_acc - adv_acc:.2f}%")

# ── Plot results ───────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(epsilons, clean_accs, "b-o", label="Clean accuracy")
plt.plot(epsilons, adv_accs,   "r-o", label="PGD adversarial accuracy")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy (%)")
plt.title(f"PGD Attack on CIFAR-10 ({num_steps} steps)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(script_dir, "pgd_cifar10_results.png"))
plt.show()
print("\nPlot saved to pgd_cifar10_results.png")

# ── Compare FGSM vs PGD ────────────────────────────────────────────────────
print("\nFGSM vs PGD comparison:")
print("PGD is stronger — lower adversarial accuracy = better attack")
print("The gap between them shows the benefit of multiple steps")

# ── Visualize example adversarial images ──────────────────────────────────
def visualize_examples(epsilon, num_steps, n=8):
    images, labels = next(iter(testloader))
    alpha      = epsilon / 4
    adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps)

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

    plt.suptitle(f"Top: original  |  Bottom: PGD adversarial (ε={epsilon}, steps={num_steps})")
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f"pgd_examples_eps{epsilon}.png"))
    plt.show()
    print(f"Example images saved.")

# Show examples at epsilon = 0.08
visualize_examples(0.08, num_steps)