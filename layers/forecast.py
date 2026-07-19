import random
import torch
import torch.nn as nn


class QueryBasedForecastingHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        T_future: int,
        N: int,
        C_metric: int,
        num_cross_attn_layers: int = 2,
    ):
        super().__init__()

        self.metric_query = nn.Parameter(torch.randn(1, T_future, N, d_model))

        cross_attn_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.cross_attention_extractor = nn.TransformerDecoder(
            cross_attn_layer, num_layers=num_cross_attn_layers
        )

        self.output_projection = nn.Linear(d_model, C_metric)

    def forward(self, fuse_out_btnd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fuse_out_btnd (torch.Tensor): The decoded multi-modal representation.
                                          Shape: [B, T_future, N, D].
        """
        B, T, N, D = fuse_out_btnd.shape

        # --- Step 1: Prepare query and key/value tensors ---
        # Expand learnable metric queries to the current batch size.
        query = self.metric_query.expand(B, -1, -1, -1)

        # Merge batch and node dimensions while preserving the temporal axis:
        # [B, T, N, D] -> [B, N, T, D] -> [B*N, T, D]
        query_reshaped = query.permute(0, 2, 1, 3).reshape(B * N, T, D)
        context_reshaped = fuse_out_btnd.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # --- Step 2: Extract temporal information with cross-attention ---
        # Cross-attention runs over B*N independent length-T node sequences.
        # tgt (target): query_reshaped
        # memory: context_reshaped
        # extracted_info_reshaped: [B*N, T, D]
        extracted_info_reshaped = self.cross_attention_extractor(
            tgt=query_reshaped, memory=context_reshaped
        )

        # --- Step 3: Project to metric space ---
        # predicted_metrics_reshaped: [B*N, T, C_metric]
        predicted_metrics_reshaped = self.output_projection(extracted_info_reshaped)

        # --- Step 4: Restore BTNC layout ---
        # [B*N, T, C_metric] -> [B, N, T, C_metric] -> [B, T, N, C_metric]
        predicted_metrics = predicted_metrics_reshaped.view(B, N, T, -1).permute(
            0, 2, 1, 3
        )

        return predicted_metrics


class ForecastingHead(nn.Module):
    def __init__(
        self,
        d_node: int,
        d_hidden: int,
        T_future: int,
        C_metric: int,
        num_gru_layers: int = 2,
    ):
        """
        A forecasting head that operates on node-wise representations [B, N, D].
        It generates future time series for each node in parallel.

        Args:
            d_node (int): Dimension of the input node representation (D).
            d_hidden (int): Dimension of the GRU hidden state.
            T_future (int): Number of future time steps to predict.
            C_metric (int): Number of metric features to predict for each node.
            num_gru_layers (int): Number of layers in the GRU.
        """
        super().__init__()
        self.T_future = T_future
        self.C_metric = C_metric
        self.num_gru_layers = num_gru_layers
        self.d_hidden = d_hidden

        # Project node representations to the GRU hidden dimension.
        self.input_proj = nn.Linear(d_node, d_hidden)

        # GRU generator produces future metric sequences per node.
        self.temporal_generator = nn.GRU(
            input_size=d_hidden,
            hidden_size=d_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Map GRU hidden states to metric channels.
        self.output_projection = nn.Linear(d_hidden, C_metric)

        # Learnable start token that drives the autoregressive-style generator.
        self.start_token = nn.Parameter(torch.randn(1, 1, d_hidden))

    def forward(self, z_node_bnd: torch.Tensor) -> torch.Tensor:
        # z_node_bnd: [B, N, D] node-level system representation.
        B, N, D = z_node_bnd.shape

        # --- Step 1: Prepare the GRU initial hidden state h_0 ---
        # [B, N, D] -> [B, N, D_hidden]
        projected_nodes = self.input_proj(z_node_bnd)

        # Merge batch and node dimensions for parallel GRU generation.
        # [B, N, D_hidden] -> [B*N, D_hidden]
        h_0_flat = projected_nodes.reshape(B * N, self.d_hidden)

        # Expand to match GRU layers:
        # [B*N, D_hidden] -> [1, B*N, D_hidden] -> [num_layers, B*N, D_hidden]
        h_0 = h_0_flat.unsqueeze(0).repeat(self.num_gru_layers, 1, 1)

        # --- Step 2: Prepare the GRU input sequence ---
        # Reuse the learnable start token across all forecast steps.
        # [1, 1, D_hidden] -> [B*N, T_future, D_hidden]
        gru_input = self.start_token.expand(B * N, self.T_future, -1)

        # --- Step 3: Generate temporal outputs with the GRU ---
        # temporal_output: [B*N, T_future, D_hidden]
        temporal_output, _ = self.temporal_generator(gru_input, h_0)

        # --- Step 4: Project to metric channels ---
        # predicted_metrics_flat: [B*N, T_future, C_metric]
        predicted_metrics_flat = self.output_projection(temporal_output)

        # --- Step 5: Restore BTNC layout ---
        # [B*N, T_future, C_metric] -> [B, N, T_future, C_metric]
        predicted_metrics = predicted_metrics_flat.view(
            B, N, self.T_future, self.C_metric
        )
        # -> [B, T_future, N, C_metric]
        predicted_metrics = predicted_metrics.permute(0, 2, 1, 3)

        return predicted_metrics


class Seq2SeqForecastingHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        T_future: int,
        C_metric: int,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        teacher_forcing_ratio: float = 0.5,
    ):
        """
        A Seq2Seq forecasting head that operates on full spatio-temporal representations [B, T, N, D].
        It uses a GRU-based encoder-decoder architecture.

        Args:
            d_model (int): Dimension of the input representation (D).
            d_hidden (int): Dimension of the GRU hidden state.
            T_future (int): Number of future time steps to predict.
            C_metric (int): Number of metric features to predict for each node.
            num_encoder_layers (int): Number of layers in the encoder GRU.
            num_decoder_layers (int): Number of layers in the decoder GRU.
            teacher_forcing_ratio (float): The probability of using teacher forcing during training.
        """
        super().__init__()
        self.T_future = T_future
        self.C_metric = C_metric
        self.d_hidden = d_hidden
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # --- 1. Encoder Part ---
        # The encoder compresses the input sequence [B*N, T, D] into a context vector.
        # Project each node-time representation before GRU encoding.
        self.input_proj = nn.Linear(d_model, d_hidden)

        self.encoder_gru = nn.GRU(
            input_size=d_hidden,
            hidden_size=d_hidden,
            num_layers=num_encoder_layers,
            batch_first=True,
        )

        # --- 2. Decoder Part ---
        # The decoder takes the context vector and generates the future sequence.
        self.decoder_gru = nn.GRU(
            input_size=d_hidden,  # Decoder input can be the previously generated output
            hidden_size=d_hidden,
            num_layers=num_decoder_layers,
            batch_first=True,
        )

        # 3. Output Projection Layer
        # Maps the decoder's hidden state to the desired metric dimension.
        self.output_projection = nn.Linear(d_hidden, C_metric)

        # A simple layer to bridge encoder and decoder hidden states if dimensions differ
        # Here they are the same, but it's good practice
        self.bridge = nn.Linear(d_hidden, d_hidden)

    def forward(
        self, x_btnd: torch.Tensor, y_btnd_metrics: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_btnd (torch.Tensor): Input tensor of shape [B, T_history, N, D].
            y_btnd_metrics (torch.Tensor, optional): Ground truth future metrics, of shape
                [B, T_future, N, C_metric]. Required for teacher forcing during training.
                If None, operates in inference mode.
        """
        B, T_history, N, D = x_btnd.shape

        # --- 1. Reshape for parallel processing over nodes ---
        # [B, T, N, D] -> [B*N, T, D]
        x_flat = x_btnd.permute(0, 2, 1, 3).reshape(B * N, T_history, D)

        # --- 2. Encoder ---
        x_proj = self.input_proj(x_flat)  # [B*N, T, D_hidden]
        _, encoder_hidden = self.encoder_gru(x_proj)
        # encoder_hidden shape: [num_layers, B*N, D_hidden]

        # --- 3. Decoder ---
        # The encoder's final hidden state is the decoder's initial hidden state.
        decoder_hidden = self.bridge(encoder_hidden)

        # Prepare storage for the outputs
        outputs = torch.zeros(B * N, self.T_future, self.C_metric, device=x_btnd.device)

        # Prepare the first input to the decoder.
        # This could be a zero tensor, a learned start token, or the last input from the encoder sequence.
        # Let's use the last hidden state projected to the input dimension.
        # A simpler way is to use a learnable start token. Let's start with a zero tensor.
        decoder_input = torch.zeros(B * N, 1, self.d_hidden, device=x_btnd.device)

        # --- 4. Autoregressive Generation Loop ---
        for t in range(self.T_future):
            # decoder_output: [B*N, 1, D_hidden], decoder_hidden: [num_layers, B*N, D_hidden]
            decoder_output, decoder_hidden = self.decoder_gru(
                decoder_input, decoder_hidden
            )

            # Project the output to the metric space
            # prediction: [B*N, 1, C_metric]
            prediction = self.output_projection(decoder_output)

            # Store the prediction
            outputs[:, t, :] = prediction.squeeze(1)

            # Decide whether to use teacher forcing
            use_teacher_forcing = (
                self.training
                and (random.random() < self.teacher_forcing_ratio)
                and (y_btnd_metrics is not None)
            )

            if use_teacher_forcing:
                # Use the ground truth as the next input.
                # We need to project it back to the hidden dimension.
                # This requires an input embedding layer for the decoder.
                # A simpler approach is to use the predicted output as the next input, even with teacher forcing.
                # Let's simplify: the *input* to the next step is always the *output* of the previous GRU step.
                decoder_input = (
                    decoder_output  # Use previous hidden state as next input
                )
            else:
                # Use the model's own prediction as the next input.
                # We need to project the C_metric output back to D_hidden.
                # This makes the model more complex.
                # The simplest Seq2Seq uses the GRU output directly.
                decoder_input = decoder_output

        # --- 5. Reshape the final output ---
        # [B*N, T_future, C_metric] -> [B, N, T_future, C_metric]
        outputs_reshaped = outputs.view(B, N, self.T_future, self.C_metric)
        # -> [B, T_future, N, C_metric]
        final_output = outputs_reshaped.permute(0, 2, 1, 3)

        return final_output
