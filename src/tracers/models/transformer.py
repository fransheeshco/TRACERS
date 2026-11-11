import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model, max_len):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x) :
        # get T of (B, T, D) with x.size(1)
        # truncate only up to passed T
        assert x.size(1) <= self.max_len, (
            f"Sequence length {x.size(1)} exceeds positional encoding max_len {self.max_len}"
        )
        x = x + self.pe[:, :x.size(1), :]
        return x


class TracerModel(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.bp_emb_dim = hyperparams['bp_emb_dim']
        self.bn_emb_dim = hyperparams['bn_emb_dim']
        self.diff_emb_dim = hyperparams['diff_emb_dim']
        self.np_hidden_size = hyperparams['np_hidden_size']
        self.np_num_layers = hyperparams['np_num_layers']
        self.ns_pre_proj_size = hyperparams['ns_pre_proj_size']
        self.ns_hidden_size = hyperparams['ns_hidden_size']
        self.ns_num_layers = hyperparams['ns_num_layers']
        self.action_emb_dim = hyperparams['action_emb_dim']
        
        # Pull max_len from hyperparams (defaults to 1000 if not provided)
        self.max_len = hyperparams.get('max_len', 1000)
        self.num_tokens = 256

        self.gelu = nn.GELU()
        # features
        self.conv_flatten_dim = 320 # hardcoded value for dim output of our stack
        self.stack = nn.Sequential(
            nn.Conv2d(3, 8, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.beat_phase_emb = nn.Embedding(49, self.bp_emb_dim)
        self.beat_num_emb = nn.Embedding(4, self.bn_emb_dim)
        self.difficulty_proj = nn.Sequential(
            nn.Linear(1, self.diff_emb_dim),
            nn.ReLU(),
            nn.Linear(self.diff_emb_dim, self.diff_emb_dim),
            nn.ReLU()
        )
        nhead = 4
        self.action_emb = nn.Embedding(self.num_tokens, self.action_emb_dim)


        # ---- NP COMPONENTS
        self.sigmoid = nn.Sigmoid()

        self.concat_np_d_model = (
            self.conv_flatten_dim
            + self.bp_emb_dim
            + self.bn_emb_dim
            + self.diff_emb_dim
        )
        assert self.concat_np_d_model % nhead == 0


        self.pe_np_encoder = PositionalEncoding(d_model=self.concat_np_d_model, max_len=self.max_len)
        np_layer = nn.TransformerEncoderLayer(
            d_model=self.concat_np_d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.np_encoder = nn.TransformerEncoder(
            np_layer,
            num_layers=self.np_num_layers
        )
        # output of transformers we project through these LL
         
        self.np_proj_1 = nn.Linear(self.concat_np_d_model, 128) 
        # output will be passed to np_proj_2 and ns branch

        self.np_proj_2 = nn.Linear(128, 1)
        # sigmoid output to get final np_pred

        # ---- NS COMPONENTS

        self.ns_pre_proj =  nn.Linear(128, self.ns_pre_proj_size)
        # output will be passed to transformer

        self.concat_ns_d_model = (
            self.concat_np_d_model
            + self.ns_pre_proj_size
            + self.action_emb_dim
        )
        assert self.concat_ns_d_model % nhead == 0

        self.pe_ns_decoder = PositionalEncoding(d_model=self.concat_ns_d_model, max_len=self.max_len)
        ns_layer = nn.TransformerEncoderLayer(
            d_model=self.concat_ns_d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.ns_decoder = nn.TransformerEncoder(
            ns_layer,
            num_layers=self.ns_num_layers
        )

        # output of transformers will project through these LL

        self.ns_proj_1 = nn.Linear(self.concat_ns_d_model, self.ns_hidden_size)
        # gelu
        self.ns_proj_2 = nn.Linear(self.ns_hidden_size, self.num_tokens)
        # raw logits


    def np_forward(self, specs, beat_phases, beat_nums, difficulties, padding_mask):
        conv_outs = self.stack(specs)
        conv_outs = conv_outs.permute(0, 2, 1, 3).flatten(2, 3) # time first, flattened feature layout

        bp_emb = self.beat_phase_emb(beat_phases)
        bn_emb = self.beat_num_emb(beat_nums)
        diff_proj = self.difficulty_proj(difficulties)

        # ========== Note Placement ========== #
        np_in = torch.cat([conv_outs, bp_emb, bn_emb, diff_proj], dim=-1)
        np_in = self.pe_np_encoder(np_in)
        np_out = self.np_encoder(np_in,
                                 src_key_padding_mask=padding_mask)
        np_proj_1_out = self.gelu(self.np_proj_1(np_out))
        np_pred = self.sigmoid(self.np_proj_2(np_proj_1_out)).squeeze(-1)

        return conv_outs, bp_emb, bn_emb, diff_proj, np_proj_1_out, np_pred


    def forward(self, specs, beat_phases, beat_nums, difficulties, actions, valid_frames):
        padding_mask = None

        if valid_frames is not None:
            padding_mask = ~valid_frames.bool()

        conv_outs, bp_emb, bn_emb, diff_proj, np_proj_1_out, np_pred = self.np_forward(
            specs, beat_phases, beat_nums, difficulties, padding_mask)

        # ========== Note Selection ========== #
        ns_pre_proj = self.gelu(self.ns_pre_proj(np_proj_1_out))
        action_emb = self.action_emb(actions)
        ns_in = torch.cat(
            [conv_outs, ns_pre_proj, bp_emb, bn_emb, diff_proj, action_emb], dim=-1)

        ns_in = self.pe_ns_decoder(ns_in)

        seq_len = ns_in.size(1)
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=ns_in.device
            ),
            diagonal=1
        )
        ns_out = self.ns_decoder(
            ns_in,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        ns_proj_1_out = self.gelu(self.ns_proj_1(ns_out))
        ns_logit = self.ns_proj_2(ns_proj_1_out)

        return np_pred, ns_logit


class TracerNpModel(TracerModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        # =====================================
        # EXPLICIT NP PROBABILITY CONDITIONING
        # =====================================

        self.np_prob_emb_dim = hyperparams.get(
            "np_prob_emb_dim", 16
        )

        self.np_prob_proj = nn.Sequential(
            nn.Linear(1, self.np_prob_emb_dim),
            nn.GELU(),
            nn.Linear(self.np_prob_emb_dim, self.np_prob_emb_dim),
            nn.GELU()
        )

        # =====================================
        # UPDATED NS COMPONENTS
        # =====================================

        nhead = 4

        # Original NS input + explicit NP probability embedding.
        self.concat_ns_d_model += self.np_prob_emb_dim

        assert self.concat_ns_d_model % nhead == 0, (
            "NS input dimension must be divisible by nhead."
        )

        # Uses the inherited self.max_len parameter
        self.pe_ns_decoder = PositionalEncoding(
            d_model=self.concat_ns_d_model,
            max_len=self.max_len
        )

        ns_layer = nn.TransformerEncoderLayer(
            d_model=self.concat_ns_d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.ns_decoder = nn.TransformerEncoder(
            ns_layer,
            num_layers=self.ns_num_layers
        )

        self.ns_proj_1 = nn.Linear(
            self.concat_ns_d_model,
            self.ns_hidden_size
        )

        self.ns_proj_2 = nn.Linear(
            self.ns_hidden_size,
            self.num_tokens
        )

    def forward(
        self,
        specs,
        beat_phases,
        beat_nums,
        difficulties,
        actions,
        valid_frames
    ):
        padding_mask = None

        if valid_frames is not None:
            padding_mask = ~valid_frames.bool()

        # =====================================
        # NP BRANCH
        # =====================================

        (
            conv_outs,
            bp_emb,
            bn_emb,
            diff_proj,
            np_proj_1_out,
            np_pred
        ) = self.np_forward(
            specs,
            beat_phases,
            beat_nums,
            difficulties,
            padding_mask
        )

        # =====================================
        # NS CONDITIONING
        # =====================================

        # Existing intermediate NP representation.
        ns_pre_proj = self.gelu(
            self.ns_pre_proj(np_proj_1_out)
        )

        # New explicit NP probability representation.
        np_prob_proj = self.np_prob_proj(
            np_pred.unsqueeze(-1)
        )

        action_emb = self.action_emb(actions)

        ns_in = torch.cat(
            [
                conv_outs,
                ns_pre_proj,
                np_prob_proj,
                bp_emb,
                bn_emb,
                diff_proj,
                action_emb
            ],
            dim=-1
        )

        # =====================================
        # NS TRANSFORMER
        # =====================================

        ns_in = self.pe_ns_decoder(ns_in)

        seq_len = ns_in.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=ns_in.device
            ),
            diagonal=1
        )

        ns_out = self.ns_decoder(
            ns_in,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        ns_proj_1_out = self.gelu(
            self.ns_proj_1(ns_out)
        )

        ns_logits = self.ns_proj_2(ns_proj_1_out)

        return np_pred, ns_logits



class _GlobalContextMixin:
    """Adds full-song audio context to an existing Tracer model."""

    def _init_global_context(self, hyperparams):
        self.global_dim = hyperparams.get("global_dim", 128)
        self.global_tokens = hyperparams.get("global_tokens", 512)
        global_heads = hyperparams.get("global_heads", 4)
        global_layers = hyperparams.get("global_num_layers", 2)
        global_ff_dim = hyperparams.get("global_ff_dim", 256)
        global_dropout = hyperparams.get("global_dropout", 0.1)

        assert self.global_dim % global_heads == 0

        # =====================================
        # GLOBAL AUDIO ENCODER
        # =====================================

        self.global_input_proj = nn.Sequential(
            nn.Linear(self.conv_flatten_dim, self.global_dim),
            nn.GELU()
        )

        self.pe_global = PositionalEncoding(
            d_model=self.global_dim,
            max_len=self.global_tokens
        )

        global_layer = nn.TransformerEncoderLayer(
            d_model=self.global_dim,
            nhead=global_heads,
            dim_feedforward=global_ff_dim,
            dropout=global_dropout,
            batch_first=True,
            activation="gelu"
        )

        self.global_encoder = nn.TransformerEncoder(
            global_layer,
            num_layers=global_layers
        )

        # =====================================
        # EXPAND NP INPUT DIMENSION
        # =====================================

        np_intermediate_dim = self.np_proj_1.out_features

        self.concat_np_d_model += self.global_dim

        assert self.concat_np_d_model % 4 == 0

        self.pe_np_encoder = PositionalEncoding(
            d_model=self.concat_np_d_model,
            max_len=self.max_len
        )

        np_layer = nn.TransformerEncoderLayer(
            d_model=self.concat_np_d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.np_encoder = nn.TransformerEncoder(
            np_layer,
            num_layers=self.np_num_layers
        )

        self.np_proj_1 = nn.Linear(
            self.concat_np_d_model,
            np_intermediate_dim
        )

        # =====================================
        # EXPAND NS INPUT DIMENSION
        # =====================================

        # This also works for TracerNpModel because its
        # NP probability embedding was already included.
        self.concat_ns_d_model += self.global_dim

        assert self.concat_ns_d_model % 4 == 0

        self.pe_ns_decoder = PositionalEncoding(
            d_model=self.concat_ns_d_model,
            max_len=self.max_len
        )

        ns_layer = nn.TransformerEncoderLayer(
            d_model=self.concat_ns_d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.ns_decoder = nn.TransformerEncoder(
            ns_layer,
            num_layers=self.ns_num_layers
        )

        self.ns_proj_1 = nn.Linear(
            self.concat_ns_d_model,
            self.ns_hidden_size
        )

    def prepare_global_context(
        self,
        global_specs,
        global_indices,
        start_frames,
        seq_len,
        device,
        dtype
    ):
        """
        Encode each unique full song, then extract the corresponding
        crop context for every sample in the batch.

        global_specs:
            List of unique full-song tensors, each [3, T_song, 80].

        global_indices:
            [B] mapping each sample to an entry in global_specs.

        start_frames:
            [B] crop positions in the original chart timeline.

        Returns:
            [B, seq_len, global_dim]
        """

        if global_specs is None or global_indices is None or start_frames is None:
            raise ValueError(
                "Global model requires global_specs, global_indices, "
                "and start_frames."
            )

        # Encode each unique song only once per forward pass.
        full_contexts = []

        for song_spec in global_specs:
            if song_spec.ndim != 3 or song_spec.shape[0] != 3:
                raise ValueError(
                    f"Expected full-song specs [3,T,80], got {tuple(song_spec.shape)}"
                )

            song_spec = song_spec.to(
                device=device,
                dtype=dtype,
                non_blocking=True
            )

            full_len = song_spec.size(1)

            if full_len < 1:
                raise ValueError("Full-song spectrogram cannot be empty.")

            # Reuse the existing CNN stack.
            features = self.stack(song_spec.unsqueeze(0))

            # [1, 64, T, 5] -> [1, T, 320]
            features = features.permute(0, 2, 1, 3).flatten(2, 3)

            # Avoid expanding very short songs to more tokens than frames.
            token_count = min(self.global_tokens, full_len)

            # [1, T, 320] -> [1, 320, T] -> [1, 512, 320]
            pooled = F.interpolate(
                features.transpose(1, 2),
                size=token_count,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

            global_in = self.global_input_proj(pooled)
            global_in = self.pe_global(global_in)

            # Unrestricted attention: the whole song is visible.
            global_out = self.global_encoder(global_in)

            # Restore a time-aligned representation.
            # [1, tokens, D] -> [1, D, tokens] -> [1, D, T]
            full_context = F.interpolate(
                global_out.transpose(1, 2),
                size=full_len,
                mode="linear",
                align_corners=False
            )

            # [T, D]
            full_context = full_context.squeeze(0).transpose(0, 1)

            full_contexts.append(full_context)

        indices = torch.as_tensor(
            global_indices
        ).detach().cpu().reshape(-1).tolist()

        starts = torch.as_tensor(
            start_frames
        ).detach().cpu().reshape(-1).tolist()

        if len(indices) != len(starts):
            raise ValueError("global_indices and start_frames must have equal lengths.")

        # Extract the corresponding section for each local crop.
        crop_contexts = []

        for song_idx, start in zip(indices, starts):
            if song_idx < 0 or song_idx >= len(full_contexts):
                raise IndexError(f"Invalid global song index: {song_idx}")

            context = full_contexts[song_idx]
            start = int(start)

            if start < 0 or start >= context.size(0):
                raise IndexError(
                    f"Crop start {start} is outside song length {context.size(0)}."
                )

            crop = context[start : start + seq_len]

            # Right padding for crops near the end of a song.
            if crop.size(0) < seq_len:
                crop = F.pad(
                    crop,
                    (0, 0, 0, seq_len - crop.size(0))
                )

            crop_contexts.append(crop)

        return torch.stack(crop_contexts, dim=0)

    def _resolve_global_context(
        self,
        specs,
        global_specs,
        global_indices,
        start_frames,
        global_context
    ):
        if global_context is not None:
            return global_context

        return self.prepare_global_context(
            global_specs=global_specs,
            global_indices=global_indices,
            start_frames=start_frames,
            seq_len=specs.size(2),
            device=specs.device,
            dtype=specs.dtype
        )

    def _np_forward_global(
        self,
        specs,
        beat_phases,
        beat_nums,
        difficulties,
        padding_mask,
        global_context
    ):
        # =====================================
        # LOCAL AUDIO FEATURES
        # =====================================

        conv_outs = self.stack(specs)
        conv_outs = conv_outs.permute(0, 2, 1, 3).flatten(2, 3)

        bp_emb = self.beat_phase_emb(beat_phases)
        bn_emb = self.beat_num_emb(beat_nums)
        diff_proj = self.difficulty_proj(difficulties)

        # =====================================
        # NP WITH GLOBAL CONTEXT
        # =====================================

        np_in = torch.cat(
            [
                conv_outs,
                global_context,
                bp_emb,
                bn_emb,
                diff_proj
            ],
            dim=-1
        )

        np_in = self.pe_np_encoder(np_in)

        np_out = self.np_encoder(
            np_in,
            src_key_padding_mask=padding_mask
        )

        np_proj_1_out = self.gelu(self.np_proj_1(np_out))
        np_pred = self.sigmoid(
            self.np_proj_2(np_proj_1_out)
        ).squeeze(-1)

        return conv_outs, bp_emb, bn_emb, diff_proj, np_proj_1_out, np_pred


class TracerGlobalModel(_GlobalContextMixin, TracerModel):
    """TracerModel with full-song global audio context."""

    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self._init_global_context(hyperparams)

    def forward(
        self,
        specs,
        beat_phases,
        beat_nums,
        difficulties,
        actions,
        valid_frames,
        global_specs=None,
        global_indices=None,
        start_frames=None,
        global_context=None
    ):
        padding_mask = None

        if valid_frames is not None:
            padding_mask = ~valid_frames.bool()

        global_context = self._resolve_global_context(
            specs,
            global_specs,
            global_indices,
            start_frames,
            global_context
        )

        (
            conv_outs,
            bp_emb,
            bn_emb,
            diff_proj,
            np_proj_1_out,
            np_pred
        ) = self._np_forward_global(
            specs,
            beat_phases,
            beat_nums,
            difficulties,
            padding_mask,
            global_context
        )

        # =====================================
        # NS WITH GLOBAL CONTEXT
        # =====================================

        ns_pre_proj = self.gelu(
            self.ns_pre_proj(np_proj_1_out)
        )

        action_emb = self.action_emb(actions)

        ns_in = torch.cat(
            [
                conv_outs,
                ns_pre_proj,
                global_context,
                bp_emb,
                bn_emb,
                diff_proj,
                action_emb
            ],
            dim=-1
        )

        ns_in = self.pe_ns_decoder(ns_in)

        seq_len = ns_in.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=ns_in.device
            ),
            diagonal=1
        )

        ns_out = self.ns_decoder(
            ns_in,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        ns_proj_1_out = self.gelu(self.ns_proj_1(ns_out))
        ns_logits = self.ns_proj_2(ns_proj_1_out)

        return np_pred, ns_logits


class TracerGlobalNpModel(_GlobalContextMixin, TracerNpModel):
    """TracerNpModel with full-song global audio context."""

    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self._init_global_context(hyperparams)

    def forward(
        self,
        specs,
        beat_phases,
        beat_nums,
        difficulties,
        actions,
        valid_frames,
        global_specs=None,
        global_indices=None,
        start_frames=None,
        global_context=None
    ):
        padding_mask = None

        if valid_frames is not None:
            padding_mask = ~valid_frames.bool()

        global_context = self._resolve_global_context(
            specs,
            global_specs,
            global_indices,
            start_frames,
            global_context
        )

        (
            conv_outs,
            bp_emb,
            bn_emb,
            diff_proj,
            np_proj_1_out,
            np_pred
        ) = self._np_forward_global(
            specs,
            beat_phases,
            beat_nums,
            difficulties,
            padding_mask,
            global_context
        )

        # =====================================
        # NS CONDITIONING
        # =====================================

        ns_pre_proj = self.gelu(
            self.ns_pre_proj(np_proj_1_out)
        )

        np_prob_proj = self.np_prob_proj(
            np_pred.unsqueeze(-1)
        )

        action_emb = self.action_emb(actions)

        ns_in = torch.cat(
            [
                conv_outs,
                ns_pre_proj,
                global_context,
                np_prob_proj,
                bp_emb,
                bn_emb,
                diff_proj,
                action_emb
            ],
            dim=-1
        )

        ns_in = self.pe_ns_decoder(ns_in)

        seq_len = ns_in.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=ns_in.device
            ),
            diagonal=1
        )

        ns_out = self.ns_decoder(
            ns_in,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        ns_proj_1_out = self.gelu(self.ns_proj_1(ns_out))
        ns_logits = self.ns_proj_2(ns_proj_1_out)

        return np_pred, ns_logits
