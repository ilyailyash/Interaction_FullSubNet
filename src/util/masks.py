import torch


def get_wav_signal(mask, length, noisy_complex, istft):
    lim = 9.9
    mask = lim * (mask >= lim) - lim * (mask <= -lim) + mask * (torch.abs(mask) < lim)
    mask = -10 * torch.log((10 - mask) / (10 + mask))

    enhanced_real = mask[..., 0] * noisy_complex[..., 0] - mask[..., 1] * noisy_complex[..., 1]
    enhanced_imag = mask[..., 1] * noisy_complex[..., 0] + mask[..., 0] * noisy_complex[..., 1]
    enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)

    signal = istft(enhanced_complex, length=length, use_mag_phase=False)
    return signal
