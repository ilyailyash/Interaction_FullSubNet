import torch


def negative_si_sdr(reference, estimation):
    estimation, reference = torch.broadcast_tensors(estimation, reference)

    reference_energy = torch.sum(reference ** 2, dim=-1, keepdim=True)

    optimal_scaling = torch.sum(reference * estimation, dim=-1, keepdim=True) \
                      / reference_energy

    projection = optimal_scaling * reference

    noise = estimation - projection

    ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim=-1)
    return -10 * torch.log10(ratio)


l1_loss = torch.nn.L1Loss
mse_loss = torch.nn.MSELoss
si_sdr_loss = negative_si_sdr
