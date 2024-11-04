import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28 * 28
hidden_sizes = [128, 64, 32]
num_epochs = 10
batch_size = 64
learning_rate = 0.001


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, _ in data_loader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return avg_loss


def get_data_loaders(batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader


def main():
    train_loader = get_data_loaders(batch_size)

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    final_loss = train(
        model,
        train_loader,
        criterion,
        optimizer,
        num_epochs,
    )

    print("Final Loss:", final_loss)


if __name__ == "__main__":
    main()
