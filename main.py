from CountNet import CountNet3D
from Dataset import ExampleDataset
from torch import nn, optim
from torch.utils import Dataloader


def main():
    geometry_dict = {
        "class_1": 0,
        "class_2": 1,
        "class_3": 2,
        "class_4": 3,
        "class_5": 4
    }
    num_classes = 5
    model = CountNet3D(num_classes, geometry_dict)

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
