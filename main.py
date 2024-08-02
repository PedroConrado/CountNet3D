from CountNet3D.CountNet import CountNet3D
from CountNet3D.Dataset import ExampleDataset
from torch import nn, optim
from torch.utils import Dataloader


def main():
    """
    Main function to train the CountNet3D model using example data.

    - Initializes the model with example geometry dictionary and camera properties.
    - Sets up the synthetic dataset and dataloader for training.
    - Trains the model for a specified number of epochs, printing the loss at each epoch.
    """

    #example dict
    geometry_dict = {
        "class_1": 0,
        "class_2": 1,
        "class_3": 2,
        "class_4": 3,
        "class_5": 4
    }
    num_classes = 5
    camera_properties = (500, 500, 500, 500)
    model = CountNet3D(num_classes, geometry_dict, camera_properties)

    dataset = ExampleDataset(num_samples=100)
    dataloader = Dataloader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for images, point_clouds, counts in dataloader:
            optimizer.zero_grad()
            outputs = model(images, point_clouds)
            loss = criterion(outputs.squeeze(), counts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")  # noqa: T201

if __name__ == "__main__":
    main()
