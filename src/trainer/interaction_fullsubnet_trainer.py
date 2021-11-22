import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.common.trainer import BaseTrainer
from src.util.acoustic_utils import mag_phase, get_complex_ideal_ratio_mask, drop_sub_band

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            dist,
            rank,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader
    ):
        super(Trainer, self).__init__(dist, rank, config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0
        speech_loss_total = 0.0
        noise_loss_total = 0.0

        for noisy, clean, noise in tqdm(self.train_dataloader, desc=f"Training {self.rank}"):
            self.optimizer.zero_grad()

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)
            noise = noise.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            noise_complex = self.torch_stft(noise)
            clean_complex = self.torch_stft(clean)

            noisy_mag, _ = mag_phase(noisy_complex)

            speech_ground_truth_cIRM = get_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
            speech_ground_truth_cIRM = drop_sub_band(
                speech_ground_truth_cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                self.model.module.num_sub_batches
            ).permute(0, 2, 3, 1)

            noise_ground_truth_cIRM = get_complex_ideal_ratio_mask(noisy_complex, noise_complex)  # [B, F, T, 2]
            noise_ground_truth_cIRM = drop_sub_band(
                noise_ground_truth_cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                self.model.module.num_sub_batches
            ).permute(0, 2, 3, 1)

            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
                noisy_mag = noisy_mag.unsqueeze(1)
                speech_pred_cRM, noise_pred_cRM = self.model(noisy_mag)
                speech_pred_cRM, noise_pred_cRM = speech_pred_cRM.permute(0, 2, 3, 1), noise_pred_cRM.permute(0, 2, 3, 1)
                speech_loss = self.loss_function(speech_ground_truth_cIRM, speech_pred_cRM)
                noise_loss = self.loss_function(noise_ground_truth_cIRM, noise_pred_cRM)
                loss = speech_loss + noise_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()
            speech_loss_total += speech_loss.item()
            noise_loss_total += noise_loss.item()


        if self.rank == 0:
            self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)
            self.writer.add_scalar(f"Speech loss/Train", speech_loss_total / len(self.train_dataloader), epoch)
            self.writer.add_scalar(f"Noise loss/Train", noise_loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        speech_loss_total = 0.0
        noise_loss_total = 0.0
        loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        speech_loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        noise_loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        item_idx_list = {"With_reverb": 0, "No_reverb": 0, }
        noisy_y_list = {"With_reverb": [], "No_reverb": [], }
        clean_y_list = {"With_reverb": [], "No_reverb": [], }
        enhanced_y_list = {"With_reverb": [], "No_reverb": [], }
        validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}

        for i, (noisy, clean, noise, name, speech_type) in tqdm(enumerate(self.valid_dataloader), desc="Validation"):
            assert len(name) == 1, "The batch size of validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)
            noise = noise.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)
            noise_complex = self.torch_stft(noise)

            noisy_mag, _ = mag_phase(noisy_complex)
            speech_ground_truth_cIRM = get_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
            noise_ground_truth_cIRM = get_complex_ideal_ratio_mask(noisy_complex, noise_complex)

            noisy_mag = noisy_mag.unsqueeze(1)
            speech_pred_cRM, noise_pred_cRM = self.model(noisy_mag)
            speech_pred_cRM, noise_pred_cRM = speech_pred_cRM.permute(0, 2, 3, 1), noise_pred_cRM.permute(0, 2, 3, 1)

            speech_loss = self.loss_function(speech_ground_truth_cIRM, speech_pred_cRM)
            noise_loss = self.loss_function(noise_ground_truth_cIRM, noise_pred_cRM)
            loss = speech_loss + noise_loss

            lim = 9.9
            speech_pred_cRM = lim * (speech_pred_cRM >= lim) - lim * (speech_pred_cRM <= -lim) + \
                              speech_pred_cRM * (torch.abs(speech_pred_cRM) < lim)
            speech_pred_cRM = -10 * torch.log((10 - speech_pred_cRM) / (10 + speech_pred_cRM))

            noise_pred_cRM = lim * (noise_pred_cRM >= lim) - lim * (noise_pred_cRM <= -lim) + \
                       noise_pred_cRM * (torch.abs(noise_pred_cRM) < lim)
            noise_pred_cRM = -10 * torch.log((10 - noise_pred_cRM) / (10 + noise_pred_cRM))

            enhanced_real = speech_pred_cRM[..., 0] * noisy_complex[..., 0] - \
                            speech_pred_cRM[..., 1] * noisy_complex[..., 1]
            enhanced_imag = speech_pred_cRM[..., 1] * noisy_complex[..., 0] + \
                            speech_pred_cRM[..., 0] * noisy_complex[..., 1]
            enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)

            despeeched_real = noise_pred_cRM[..., 0] * noisy_complex[..., 0] - \
                              noise_pred_cRM[..., 1] * noisy_complex[..., 1]
            despeeched_imag = noise_pred_cRM[..., 1] * noisy_complex[..., 0] + \
                              noise_pred_cRM[..., 0] * noisy_complex[..., 1]
            despeeched_complex = torch.stack((despeeched_real, despeeched_imag), dim=-1)

            enhanced = self.istft(enhanced_complex, length=noisy.size(-1), use_mag_phase=False)
            despeeched = self.istft(despeeched_complex, length=noisy.size(-1), use_mag_phase=False)

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            noise = noise.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()
            despeeched = despeeched.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced) == len(despeeched)
            loss_total += loss
            speech_loss_total += speech_loss
            noise_loss_total += noise_loss

            """=== === === Visualization === === ==="""
            # Separated Loss
            loss_list[speech_type] += loss
            speech_loss_list[speech_type] += speech_loss
            noise_loss_list[speech_type] += noise_loss
            item_idx_list[speech_type] += 1

            if item_idx_list[speech_type] <= visualization_n_samples:
                self.spec_audio_visualization(noisy, enhanced, clean, name, epoch,
                                              noise=noise,
                                              despeeched=despeeched,
                                              mark=speech_type)

            noisy_y_list[speech_type].append(noisy)
            clean_y_list[speech_type].append(clean)
            enhanced_y_list[speech_type].append(enhanced)

        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)
        self.writer.add_scalar(f"Speech loss/Validation_Total", speech_loss_total / len(self.valid_dataloader), epoch)
        self.writer.add_scalar(f"Noise loss/Validation_Total", noise_loss_total / len(self.valid_dataloader), epoch)

        for speech_type in ("With_reverb", "No_reverb"):
            self.writer.add_scalar(f"Loss/{speech_type}", loss_list[speech_type] / len(self.valid_dataloader), epoch)
            self.writer.add_scalar(f"Speech loss/{speech_type}",
                                   speech_loss_list[speech_type] / len(self.valid_dataloader), epoch)
            self.writer.add_scalar(f"Noise loss/{speech_type}",
                                   noise_loss_list[speech_type] / len(self.valid_dataloader), epoch)

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type], clean_y_list[speech_type], enhanced_y_list[speech_type],
                visualization_metrics, epoch, visualization_num_workers, mark=speech_type
            )

        return validation_score_list["No_reverb"]
