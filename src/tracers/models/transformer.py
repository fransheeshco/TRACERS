import torch
import torch.nn as nn
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
