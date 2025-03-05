import torch


def create_2D_grid(x_size, y_size):
    meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
    print(meshgrid)
    # NOTE: modified
    batch_x, batch_y = torch.meshgrid(
        *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
    print(batch_x, batch_y)
    batch_x = batch_x + 0.5
    batch_y = batch_y + 0.5
    coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
    # coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
    return coord_base

create_2D_grid(16, 16)