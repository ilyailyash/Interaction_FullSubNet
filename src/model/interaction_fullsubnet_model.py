import torch
import torch.nn as nn
from torch.nn import functional

from src.common.model import BaseModel
from src.model.module.sequence import SequenceModel
from src.model.module.interaction_sequence import InteractionSequenceModel
from src.util.acoustic_utils import drop_sub_band
from src.util.interactions import Interaction


class Model(BaseModel):
    def __init__(self,
                 n_freqs,
                 n_neighbor,
                 look_ahead,
                 sequence_model,
                 num_sub_layers,
                 fband_output_activate_function,
                 sband_output_activate_function,
                 fband_model_hidden_size,
                 sband_model_hidden_size,
                 bidirectional=False,
                 weight_init=True,
                 num_sub_batches=3,
                 use_offline_norm=True,
                 use_cumulative_norm=False,
                 use_forgetting_norm=False,
                 use_hybrid_norm=False,
                 ):
        """
        FullSubNet model

        Input: [B, 1, F, T]
        Output: [B, 2, F, T]

        Args:
            n_freqs: Frequency dim of the input
            n_neighbor: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fband_model = SequenceModel(
            input_size=n_freqs,
            output_size=n_freqs,
            hidden_size=fband_model_hidden_size,
            num_layers=2,
            bidirectional=bidirectional,
            sequence_model=sequence_model,
            output_activate_function=fband_output_activate_function
        )

        for i in range(num_sub_layers):
            input_size = (n_neighbor * 2 + 1) + 1 + 2 if i == 0 else sband_model_hidden_size
            setattr(self, 'speech_sband_model_' + str(i),
                    InteractionSequenceModel(
                        input_size=input_size,
                        hidden_size=sband_model_hidden_size,
                        num_layers=1,
                        bidirectional=bidirectional,
                        sequence_model=sequence_model,
                        output_activate_function=sband_output_activate_function
                    )
                    )

            setattr(self, 'noise_sband_model_' + str(i),
                    InteractionSequenceModel(
                        input_size=input_size,
                        hidden_size=sband_model_hidden_size,
                        num_layers=1,
                        bidirectional=bidirectional,
                        sequence_model=sequence_model,
                        output_activate_function=sband_output_activate_function
                    )
                    )

            setattr(self, 'speech_interaction_' + str(i), Interaction())
            setattr(self, 'noise_interaction_' + str(i), Interaction())

        if bidirectional:
            self.speech_output_layer = nn.Linear(sband_model_hidden_size * 2, 2)
            self.noise_output_layer = nn.Linear(sband_model_hidden_size * 2, 2)
        else:
            self.speech_output_layer = nn.Linear(sband_model_hidden_size, 2)
            self.noise_output_layer = nn.Linear(sband_model_hidden_size, 2)

        self.n_neighbor = n_neighbor
        self.look_ahead = look_ahead
        self.num_sub_layers = num_sub_layers
        self.use_offline_norm = use_offline_norm
        self.use_cumulative_norm = use_cumulative_norm
        self.use_forgetting_norm = use_forgetting_norm
        self.use_hybrid_norm = use_hybrid_norm
        self.num_sub_batches = num_sub_batches

        assert (use_hybrid_norm + use_forgetting_norm + use_cumulative_norm + use_offline_norm) == 1, \
            "Only Supports one Norm method."

        if weight_init:
            self.apply(self.weight_init)


    def forward(self, input):
        """
        Args:
            input: [B, 1, F, T]

        Returns:
            [B, 2, F, T]
        """
        assert input.dim() == 4
        # Pad look ahead
        input = functional.pad(input, [0, self.look_ahead])
        batch_size, n_channels, n_freqs, n_frames = input.size()
        assert n_channels == 1, f"{self.__class__.__name__} takes mag feature as inputs."

        """=== === === Full-Band sub Model === === ==="""
        if self.use_offline_norm:
            fband_mu = torch.mean(input, dim=(1, 2, 3)).reshape(batch_size, 1, 1, 1)  # 语谱图算一个均值
            fband_input = input / (fband_mu + 1e-10)
        elif self.use_cumulative_norm:
            fband_input = self.cumulative_norm(input)
        elif self.use_forgetting_norm:
            fband_input = self.forgetting_norm(input.reshape(batch_size, n_channels * n_freqs, n_frames), 192)
            fband_input.reshape(batch_size, n_channels, n_freqs, n_frames)
        elif self.use_hybrid_norm:
            fband_input = self.hybrid_norm(input.reshape(batch_size, n_channels * n_freqs, n_frames), 192)
            fband_input.reshape(batch_size, n_channels, n_freqs, n_frames)
        else:
            raise NotImplementedError("You must set up a type of Norm. E.g., offline_norm, cumulative_norm, forgetting_norm.")

        # [B, 1, F, T] => [B, F, T] => [B, 1, F, T]
        fband_input = fband_input.reshape(batch_size, n_channels * n_freqs, n_frames)
        fband_output = self.fband_model(fband_input)
        fband_output = fband_output.reshape(batch_size, n_channels, n_freqs, n_frames)

        """=== === === Sub-Band sub Model === === ==="""
        # [B, 1, F, T] => unfold => [B, N=F, C, F_s, T] => [B * N, F_s, T]
        input_unfolded = self.unfold(input, n_neighbor=self.n_neighbor)
        fband_output_unfolded = self.unfold(fband_output, n_neighbor=1)

        input_unfolded = input_unfolded.reshape(batch_size * n_freqs, self.n_neighbor * 2 + 1, n_frames)
        fband_output_unfolded = fband_output_unfolded.reshape(batch_size * n_freqs, 2 + 1, n_frames)

        # [B * F, (F_s + 3), T]
        sband_input = torch.cat([input_unfolded, fband_output_unfolded], dim=1)

        if self.use_offline_norm:
            sband_mu = torch.mean(sband_input, dim=(1, 2)).reshape(batch_size * n_freqs, 1, 1)
            sband_input = sband_input / (sband_mu + 1e-10)
        elif self.use_cumulative_norm:
            sband_input = self.cumulative_norm(sband_input)
        elif self.use_forgetting_norm:
            sband_input = self.forgetting_norm(sband_input, 192)
        elif self.use_hybrid_norm:
            sband_input = self.hybrid_norm(sband_input, 192)
        else:
            raise NotImplementedError("You must set up a type of Norm. E.g., offline_norm, cumulative_norm, forgetting_norm.")

        # Speed up training without significant performance degradation
        # This part of the content will be updated in the paper later
        if batch_size > 1:
            sband_input = sband_input.reshape(batch_size, n_freqs, self.n_neighbor * 2 + 1 + 2 + 1, n_frames)
            sband_input = drop_sub_band(sband_input.permute(0, 2, 1, 3), num_sub_batches=self.num_sub_batches)
            n_freqs = sband_input.shape[2]
            sband_input = sband_input.permute(0, 2, 1, 3).reshape(-1, self.n_neighbor * 2 + 1 + 2 + 1, n_frames)

        sband_input = sband_input.permute(0, 2, 1).contiguous()
        speech_branch = noise_branch = sband_input
        # [B * F, (F_s + 1), T] => [B * F, 2, T] => [B, F, 2, T]

        for i in range(self.num_sub_layers):
            speech_sband_ouput = getattr(self, 'speech_sband_model_' + str(i))(speech_branch)
            noise_sband_ouput = getattr(self, 'noise_sband_model_' + str(i))(noise_branch)
            speech_branch = getattr(self, 'speech_interaction_' + str(i))(speech_sband_ouput, noise_sband_ouput)
            noise_branch = getattr(self, 'noise_interaction_' + str(i))(noise_sband_ouput, speech_sband_ouput)

        speech_mask = self.speech_output_layer(speech_branch)
        noise_mask = self.noise_output_layer(noise_branch)

        speech_mask = speech_mask.permute(0, 2, 1).contiguous()
        noise_mask = noise_mask.permute(0, 2, 1).contiguous()

        speech_mask = speech_mask.reshape(batch_size, n_freqs, 2, n_frames).permute(0, 2, 1, 3).contiguous()
        noise_mask = noise_mask.reshape(batch_size, n_freqs, 2, n_frames).permute(0, 2, 1, 3).contiguous()

        output_speech = speech_mask[:, :, :, self.look_ahead:]
        output_noise = noise_mask[:, :, :, self.look_ahead:]

        return output_speech, output_noise

