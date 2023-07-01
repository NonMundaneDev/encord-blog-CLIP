import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable


# Define the data transformations
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the datasets
train_dataset = datasets.ImageFolder(
    r'Clip_GT_labels\Train',
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    r'Clip_GT_labels\Val',
    transform=val_transforms
)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()

        # input shape (3, 256, 256)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # shape (16, 128, 128)

        # input shape (16, 128, 128)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # output shape (32, 64, 64)

        # input shape (32, 64, 64)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # output shape (64, 32, 32)

        # input shape (64, 32, 32)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # output shape (32, 16, 16)

        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Initialize the model and define the loss function and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 50
best_acc = 0.0

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0

    model.train()
    for images, labels in train_loader:
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    val_loss = 0.0
    val_acc = 0.0

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_acc / len(val_dataset)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'cnn_model.pth')
