import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Initializing DEEP Sea-Shore Network on: {device}")

# --- DATA ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# --- THE DEEP MODEL ---
class DeepSeaShore(nn.Module):
    def __init__(self, width=2000):
        super().__init__()
        # Layer 1: Pixels -> Edges
        self.fc1 = nn.Linear(784, width, bias=False)
        # Layer 2: Edges -> Shapes (The new layer)
        self.fc2 = nn.Linear(width, width, bias=False)
        # Layer 3: Shapes -> Digits (Readout)
        self.fc3 = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.normalize(x, p=2, dim=1)
        
        # Layer 1
        x = F.relu(self.fc1(x))
        # Layer 2 (Normalize input to L2 to keep physics consistent)
        x = F.normalize(x, p=2, dim=1) 
        x = F.relu(self.fc2(x))
        # Output
        x = self.fc3(x)
        return x

# Hyperparameters from your best run
WIDTH = 2000
LR = 0.2
BP_RATIO = 3 # 3 steps Backprop, 7 steps Hebbian

model = DeepSeaShore(width=WIDTH).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Helper function for Hebbian Update (Oja's Rule + Winner Take All)
def hebbian_update(layer, inputs, learning_rate):
    with torch.no_grad():
        # 1. Normalize Inputs
        waves = F.normalize(inputs, p=2, dim=1)
        weights = layer.weight
        
        # 2. Similarity & Competition
        similarity = torch.mm(waves, weights.t())
        winners = torch.argmax(similarity, dim=1)
        winner_mask = F.one_hot(winners, num_classes=weights.shape[0]).float()
        
        # 3. Sedimentation
        waves_sum = torch.mm(winner_mask.t(), waves)
        win_counts = winner_mask.sum(dim=0).unsqueeze(1) + 1e-6
        waves_avg = waves_sum / win_counts
        has_won = (win_counts > 1e-5).float()
        
        # 4. Update & Erode
        delta = learning_rate * has_won * (waves_avg - weights)
        layer.weight.add_(delta)
        layer.weight.div_(layer.weight.norm(dim=1, keepdim=True))

print(f"Training Deep Network (Width={WIDTH}, Hebbian LR={LR}, Split={BP_RATIO}/10)...")

for epoch in range(5):
    running_loss = 0.0
    bp_steps = 0
    
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        images_flat = images.view(-1, 784)
        
        cycle_idx = i % 10
        
        if cycle_idx < BP_RATIO:
            # --- BACKPROP (30%) ---
            # Updates ALL layers globally
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Renormalize Hebbian layers to keep them stable
            with torch.no_grad():
                model.fc1.weight.div_(model.fc1.weight.norm(dim=1, keepdim=True))
                model.fc2.weight.div_(model.fc2.weight.norm(dim=1, keepdim=True))
            
            running_loss += loss.item()
            bp_steps += 1
            
        else:
            # --- HEBBIAN (70%) ---
            # Updates layers LOCALLY and SEQUENTIALLY
            with torch.no_grad():
                # Update Layer 1 based on Raw Pixels
                hebbian_update(model.fc1, images_flat, LR)
                
                # Get Layer 1 Output to serve as "Waves" for Layer 2
                # Note: We must use the current weights of FC1 to generate signal
                l1_out = F.relu(model.fc1(F.normalize(images_flat, p=2, dim=1)))
                
                # Update Layer 2 based on Layer 1 Activity
                hebbian_update(model.fc2, l1_out, LR)

    print(f"Epoch {epoch+1}: Avg Supervised Loss: {running_loss/bp_steps:.4f}")

# --- FINAL EVALUATION ---
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"------------------------------------------------")
print(f"FINAL DEEP ACCURACY: {100 * correct / total:.2f}%")
print(f"------------------------------------------------")