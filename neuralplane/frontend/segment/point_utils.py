import torch


def denormalise_coordinates(x_norm, dims):
    dims = torch.as_tensor(dims, dtype=torch.float32, device=x_norm.device)
    x_pixel = 0.5 * (dims - 1) * ((x_norm) + 1 )
    return x_pixel.round().long()

def normalise_coordinates(x_pixel, dims):
    inv = 1.0 / (torch.as_tensor(dims, dtype=torch.float32, device=x_pixel.device) - 1)

    x_norm = 2 * x_pixel * inv - 1
    return x_norm