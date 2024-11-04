import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28 * 28
hidden_sizes = [128, 64, 32]


class Autoencoder(nn.Module):
    """Autoencoder model for reconstructing input images.

    This class defines a simple feedforward neural network that consists of an encoder and a decoder.

    Attributes:
        encoder (nn.Sequential): The encoder part of the autoencoder.
        decoder (nn.Sequential): The decoder part of the autoencoder.
    """

    def __init__(self, input_size, hidden_sizes):
        """Initializes the Autoencoder with specified input size and hidden layer sizes.

        Args:
            input_size (int): The size of the input images.
            hidden_sizes (list[int]): The sizes of the hidden layers.
        """
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
        """Defines the forward pass of the autoencoder.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The reconstructed output.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, data_loader, criterion, optimizer, num_epochs):
    """Trains the autoencoder model.

    Args:
        model (Autoencoder): The autoencoder model to train.
        data_loader (DataLoader): The data loader for training data.
        criterion (nn.Module): The loss function used for training.
        optimizer (optim.Optimizer): The optimizer used for updating weights.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        float: The average loss over the training dataset.
    """
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        for data, _ in data_loader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        epoch_time = time.time() - start_time
        print(f"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}")

        wandb.log(
            {
                "avg_loss": avg_loss,
                "epoch_time": epoch_time,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if (epoch + 1) % 5 == 0:
            sample_data, _ = next(iter(data_loader))
            with torch.no_grad():
                reconstructed = model(sample_data.to(device)).cpu()
            wandb.log(
                {
                    "sample_reconstruction": [
                        wandb.Image(reconstructed[i].view(28, 28).numpy()) for i in range(10)
                    ]
                }
            )

    wandb.log({"final_loss": avg_loss})
    return avg_loss


def get_data_loaders(batch_size):
    """Creates data loaders for the FashionMNIST dataset.

    Args:
        batch_size (int): The batch size for the data loader.

    Returns:
        DataLoader: The data loader for the training dataset.
    """
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


def objective(trial):
    """Defines the objective function for Optuna to optimize hyperparameters.

    Args:
        trial (optuna.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: The final loss of the trained model.
    """
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    num_epochs = trial.suggest_int("num_epochs", 5, 20)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    wandb.init(
        project="optuna-autoencoder",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
        },
    )

    train_loader = get_data_loaders(batch_size)

    model = Autoencoder(input_size=input_size, hidden_sizes=hidden_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    final_loss = train(
        model,
        train_loader,
        criterion,
        optimizer,
        num_epochs,
    )

    wandb.log({"final_loss": final_loss})
    wandb.finish()
    return final_loss


def main():
    """Main function to run the Optuna hyperparameter optimization."""
    wandb.login()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
