from __future__ import annotations
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from conditioning import ConditionConfig, FrameConditioner
from patterns import DelayPattern, default_delays

@dataclass
class ModelConfig:
    codebook_size: int = 1024
    num_codebooks: int = 4
    delays: list[int] | None = None
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 2048
    text_model: str = 'google-t5/t5-small'
    text_max_length: int = 128
    text_condition_dropout: float = 0.1
    structured_condition_dropout: float = 0.05
    audio_context_layers: int = 4
    audio_context_heads: int = 8
    condition: dict = field(default_factory=lambda: asdict(ConditionConfig()))

    def to_dict(self) -> dict:
        return asdict(self)

    def resolved_delays(self) -> tuple[int, ...]:
        if self.delays is None:
            return default_delays(self.num_codebooks)
        if len(self.delays) != self.num_codebooks:
            raise ValueError('len(delays) must match num_codebooks')
        return tuple((int(x) for x in self.delays))

class FrozenTextEncoder(nn.Module):
    def __init__(self, model_name: str, max_length: int):
        super().__init__()
        try:
            from transformers import AutoTokenizer, T5EncoderModel
        except ImportError as e:
            raise RuntimeError('Install transformers + sentencepiece to use text conditioning.') from e
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.d_model)

    @torch.no_grad()
    def forward(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoder.eval()
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        hidden = self.encoder(input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device)).last_hidden_state
        return (hidden, batch.attention_mask.to(device).bool())

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None=None) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (self._split(q), self._split(k), self._split(v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.out(self._merge(y))

    def step(self, x: torch.Tensor, cache: Optional[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k_new, v_new = self.qkv(x).chunk(3, dim=-1)
        q, k_new, v_new = (self._split(q), self._split(k_new), self._split(v_new))
        if cache is None:
            k, v = (k_new, v_new)
        else:
            k = torch.cat([cache[0], k_new], dim=2)
            v = torch.cat([cache[1], v_new], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return (self.out(self._merge(y)), (k, v))

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def prepare_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.kv(context).chunk(2, dim=-1)
        return (self._split(k), self._split(v))

    def attend_prepared(self, x: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor], context_mask: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q(x))
        k, v = kv
        mask = context_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        return self.out(self._merge(y))

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        return self.attend_prepared(x, self.prepare_kv(context), context_mask)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(d_model, hidden * 2)
        self.out_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_proj(x).chunk(2, dim=-1)
        return self.dropout(self.out_proj(F.silu(a) * b))

class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.text_attn = CrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.audio_attn = CrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln4 = nn.LayerNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.ff_mult * cfg.d_model, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor, audio_ctx: torch.Tensor | None, audio_ctx_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.ln1(x)))
        x = x + self.drop(self.text_attn(self.ln2(x), text, text_mask))
        if audio_ctx is not None and audio_ctx_mask is not None:
            x = x + self.drop(self.audio_attn(self.ln3(x), audio_ctx, audio_ctx_mask))
        x = x + self.ff(self.ln4(x))
        return x

    def step(self, x: torch.Tensor, self_cache: Optional[tuple[torch.Tensor, torch.Tensor]], text_kv: tuple[torch.Tensor, torch.Tensor], text_mask: torch.Tensor, audio_kv: tuple[torch.Tensor, torch.Tensor] | None, audio_mask: torch.Tensor | None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        y, cache = self.self_attn.step(self.ln1(x), self_cache)
        x = x + y
        x = x + self.text_attn.attend_prepared(self.ln2(x), text_kv, text_mask)
        if audio_kv is not None and audio_mask is not None:
            x = x + self.audio_attn.attend_prepared(self.ln3(x), audio_kv, audio_mask)
        x = x + self.ff(self.ln4(x))
        return (x, cache)

class AudioContextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.audio_context_heads, dim_feedforward=cfg.ff_mult * cfg.d_model, dropout=cfg.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.audio_context_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x, src_key_padding_mask=~known_mask)
        return self.norm(y)

def _duplicate_conditions(cond: dict[str, torch.Tensor], times: int) -> dict[str, torch.Tensor]:
    if times == 1:
        return cond
    return {k: v.repeat_interleave(times, dim=0) if v.ndim > 0 else v for k, v in cond.items()}

class AudioTokenTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pattern = DelayPattern(cfg.resolved_delays(), pad_id=cfg.codebook_size)
        if cfg.text_model == '__none__':
            self.text_encoder = None
            self.null_text = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.text_proj = nn.Identity()
        else:
            self.text_encoder = FrozenTextEncoder(cfg.text_model, cfg.text_max_length)
            self.null_text = None
            self.text_proj = nn.Linear(self.text_encoder.hidden_size, cfg.d_model)
        self.codebook_embeddings = nn.ModuleList([nn.Embedding(cfg.codebook_size + 1, cfg.d_model) for _ in range(cfg.num_codebooks)])
        self.bos = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.frame_conditioner = FrameConditioner(cfg.d_model, ConditionConfig(**cfg.condition))
        self.input_dropout = nn.Dropout(cfg.dropout)
        self.audio_context_encoder = AudioContextEncoder(cfg)
        self.audio_context_mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.audio_context_pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.heads = nn.ModuleList([nn.Linear(cfg.d_model, cfg.codebook_size, bias=False) for _ in range(cfg.num_codebooks)])
        self._init_trainable()

    def _init_trainable(self) -> None:
        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.audio_context_mask_token, std=0.02)
        modules = [self.text_proj, self.codebook_embeddings, self.pos, self.frame_conditioner, self.audio_context_encoder, self.audio_context_pos, self.blocks, self.final_ln, self.heads]
        for root in modules:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        meaningful = [k for k in missing if not k.startswith('text_encoder.encoder.')]
        if meaningful or unexpected:
            raise RuntimeError(f'Checkpoint mismatch. Missing={meaningful}, unexpected={unexpected}')

    def encode_text(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            b = len(texts)
            return (self.null_text.expand(b, 1, -1), torch.ones((b, 1), device=device, dtype=torch.bool))
        hidden, mask = self.text_encoder(texts, device)
        return (self.text_proj(hidden), mask)

    def embed_code_columns(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3 or codes.shape[1] != self.cfg.num_codebooks:
            raise ValueError(f'Expected [B,{self.cfg.num_codebooks},T], got {tuple(codes.shape)}')
        x = None
        for q, emb in enumerate(self.codebook_embeddings):
            e = emb(codes[:, q].clamp(0, self.cfg.codebook_size))
            x = e if x is None else x + e
        return x / math.sqrt(self.cfg.num_codebooks)

    def encode_audio_context(self, codes: torch.Tensor | None, known_mask: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if codes is None or known_mask is None:
            return (None, None)
        if codes.shape[-1] > self.cfg.max_seq_len:
            raise ValueError('audio context exceeds model max_seq_len')
        x = self.embed_code_columns(codes)
        pos = torch.arange(codes.shape[-1], device=codes.device)
        x = x + self.audio_context_pos(pos)[None]
        x = torch.where(known_mask[..., None], x, self.audio_context_mask_token.expand_as(x))
        safe_mask = known_mask.clone()
        all_missing = ~safe_mask.any(dim=1)
        if all_missing.any():
            safe_mask[all_missing, 0] = True
        return (self.audio_context_encoder(x, safe_mask), safe_mask)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, generator: torch.Generator | None) -> torch.Tensor:
        logits = logits / temperature
        vocab = logits.shape[-1]
        if 0 < top_k < vocab:
            values, _ = torch.topk(logits, k=top_k, dim=-1)
            logits = logits.masked_fill(logits < values[:, -1:], float('-inf'))
        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
            filtered = torch.full_like(logits, float('-inf'))
            filtered.scatter_(1, sorted_idx, sorted_logits)
            logits = filtered
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    @torch.no_grad()
    def generate(self, prompt: str, frames: int, cond: dict[str, torch.Tensor], *, prompt_codes: torch.Tensor | None=None, audio_context_codes: torch.Tensor | None=None, audio_context_known: torch.Tensor | None=None, temperature: float=1.0, top_k: int=250, top_p: float=0.95, cfg_scale: float=3.0, seed: int | None=None) -> torch.Tensor:
        self.eval()
        if temperature <= 0:
            raise ValueError('temperature must be > 0')
        total_steps = self.pattern.sequence_length(frames)
        if total_steps > self.cfg.max_seq_len:
            raise ValueError(f'Need {total_steps} delayed steps; max_seq_len={self.cfg.max_seq_len}. Use generate_long.py for longer music.')
        device = next(self.parameters()).device
        if prompt_codes is not None:
            prompt_codes = prompt_codes.to(device)
            if prompt_codes.ndim == 2:
                prompt_codes = prompt_codes.unsqueeze(0)
            if prompt_codes.shape[1] != self.cfg.num_codebooks:
                raise ValueError('prompt codec does not match checkpoint codebook count')
            prompt_frames = prompt_codes.shape[-1]
            if prompt_frames > frames:
                raise ValueError('prompt is longer than requested output')
        else:
            prompt_frames = 0
        use_cfg = cfg_scale != 1.0
        batch = 2 if use_cfg else 1
        texts = [prompt, ''] if use_cfg else [prompt]
        text_ctx, text_mask = self.encode_text(texts, device)
        cond = {k: v.to(device) for k, v in cond.items()}
        if next(iter(cond.values())).shape[0] != 1:
            raise ValueError('generation conditions must have batch size 1')
        cond_b = _duplicate_conditions(cond, batch)
        if audio_context_codes is not None:
            audio_context_codes = audio_context_codes.to(device)
            audio_context_known = audio_context_known.to(device)
            if audio_context_codes.ndim == 2:
                audio_context_codes = audio_context_codes.unsqueeze(0)
            if audio_context_known.ndim == 1:
                audio_context_known = audio_context_known.unsqueeze(0)
            if use_cfg:
                audio_context_codes = audio_context_codes.repeat(batch, 1, 1)
                audio_context_known = audio_context_known.repeat(batch, 1)
        audio_ctx, audio_mask = self.encode_audio_context(audio_context_codes, audio_context_known)
        text_kvs = [block.text_attn.prepare_kv(text_ctx) for block in self.blocks]
        audio_kvs = [block.audio_attn.prepare_kv(audio_ctx) if audio_ctx is not None else None for block in self.blocks]
        caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.blocks)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        delayed = torch.full((1, self.cfg.num_codebooks, total_steps), self.cfg.codebook_size, device=device, dtype=torch.long)
        for step in range(total_steps):
            if step == 0:
                x = self.bos.expand(batch, 1, -1)
            else:
                prev = delayed[:, :, step - 1:step].expand(batch, -1, -1)
                x = self.embed_code_columns(prev)
            frame_idx = torch.tensor([min(step, frames - 1)], device=device)
            x = x + self.pos.weight[step][None, None] + self.frame_conditioner(cond_b, frame_idx)
            new_caches = []
            for i, block in enumerate(self.blocks):
                x, cache = block.step(x, caches[i], text_kvs[i], text_mask, audio_kvs[i], audio_mask)
                new_caches.append(cache)
            caches = new_caches
            h = self.final_ln(x)[:, 0]
            logits = torch.stack([head(h) for head in self.heads], dim=1)
            if use_cfg:
                cond_logits, uncond_logits = (logits[0], logits[1])
                step_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            else:
                step_logits = logits[0]
            sampled = self._sample(step_logits, temperature, top_k, top_p, generator)
            for q, original_t in self.pattern.valid_codebooks_at_step(step, frames):
                if original_t < prompt_frames:
                    delayed[0, q, step] = prompt_codes[0, q, original_t]
                else:
                    delayed[0, q, step] = sampled[q]
        return self.pattern.revert(delayed, frames)
