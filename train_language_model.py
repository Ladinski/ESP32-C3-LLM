"""Train and export a tiny 1-layer character transformer using fake INT4 quantization.

The text is intentionally tiny: this is a training + deployment sanity check,
NOT evidence of general-purpose language modeling or held-out accuracy.

Run: uv run --python 3.12 --with torch --with numpy python train_language_model.py
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from character_generator import CHARS, DIM, FF, VOCAB, MAX_TOKENS, pack4, fnv32, cached_decode, unpack_model

torch.set_num_threads(1)
torch.manual_seed(20261113)
np.random.seed(20261113)
OUTPUT = Path(__file__).resolve().parent
TEXT = 'hello esp32!\n'
assert len(CHARS) == VOCAB == 16
assert len(TEXT) - 1 == MAX_TOKENS == 12


def fake_quant(tensor, scale, low, high):
    quant = (tensor * scale).round().clamp(low, high) / scale
    return tensor + (quant - tensor).detach()  # straight-through estimator


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(VOCAB, DIM) * 0.22)
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(*shape) * 0.27)
            for shape in [(DIM, DIM)] * 4 + [(FF, DIM), (DIM, FF), (VOCAB, DIM)]
        ])
        offsets = torch.arange(DIM).unsqueeze(0) % 4
        positions = torch.arange(MAX_TOKENS).unsqueeze(1) % 4
        self.register_buffer('pos', (offsets == positions).float() * 0.125)
        self.register_buffer('mask', torch.ones(MAX_TOKENS, MAX_TOKENS, dtype=torch.bool).tril())

    def forward(self, tokens):
        n = tokens.shape[0]
        embedding = fake_quant(self.embed, 8.0, -128, 127)
        weights = [fake_quant(w, 16.0, -8, 7) for w in self.w]
        wq, wk, wv, wo, w1, w2, head = weights
        x = embedding[tokens] + self.pos[:n]
        z = F.layer_norm(x, (DIM,), eps=1e-5)
        q, k, v = (F.linear(z, w) for w in (wq, wk, wv))
        scores = q @ k.T / (DIM ** 0.5)
        scores = scores.masked_fill(~self.mask[:n, :n], -1e9)
        ctx = scores.softmax(dim=-1) @ v
        residual = x + F.linear(ctx, wo)
        ff = F.linear(F.relu(F.linear(F.layer_norm(residual, (DIM,), eps=1e-5), w1)), w2)
        return F.linear(F.layer_norm(residual + ff, (DIM,), eps=1e-5), head)


def main():
    inputs = torch.tensor([CHARS.index(c) for c in TEXT[:-1]], dtype=torch.long)
    labels = torch.tensor([CHARS.index(c) for c in TEXT[1:]], dtype=torch.long)
    model = Decoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    print('Phase 11 | Quantization-aware training of a tiny character transformer')
    print('Training text:', repr(TEXT), '| train positions:', len(inputs))
    print('This tiny training set only tests learning and ESP32 deployment, not generalization.')
    best = 0
    for step in range(1, 4001):
        model.train()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        with torch.no_grad():
            correct = int((logits.argmax(dim=-1) == labels).sum().item())
            best = max(best, correct)
        if step == 1 or step % 200 == 0:
            print(f'step {step:4d} | loss {loss.item():.4f} | next-char accuracy {correct}/12')
        if correct == 12 and loss.item() < 0.025:
            print('Training objective met.')
            break
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        correct = int((logits.argmax(-1) == labels).sum().item())
        quant_embed = (model.embed * 8).round().clamp(-128, 127).to(torch.int8).cpu().numpy()
        quant_weights = [(w * 16).round().clamp(-8, 7).to(torch.int8).cpu().numpy()
                         for w in model.w]

    payload = quant_embed.tobytes() + b''.join(pack4(w) for w in quant_weights)
    assert len(payload) == 1408
    emb, ws = unpack_model(payload)
    used, predicted, scores = cached_decode(emb, ws, int(inputs[0]), MAX_TOKENS)
    generated = CHARS[int(inputs[0])] + ''.join(CHARS[i] for i in predicted)
    expected = TEXT
    print('Exported autoregressive sample:', repr(generated))
    print(f'Quantized train accuracy: {correct}/{len(inputs)}')
    if correct < len(inputs) or generated != expected:
        raise SystemExit('Training did not meet the target. Rerun training; no model written.')
    (OUTPUT / 'phase11_model.bin').write_bytes(payload)
    metrics = {'training_text': TEXT, 'train_positions': len(inputs),
               'training_accuracy': correct / len(inputs), 'generated_sample': generated,
               'quantization': 'weights int4, embedding int8, float activations',
               'model_bytes': len(payload), 'fnv32': f'{fnv32(payload):08x}',
               'generalization_tested': False}
    (OUTPUT / 'phase11_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'SAVED phase11_model.bin ({len(payload)} bytes, hash {fnv32(payload):08x})')
    print('TRAINING PASS')


if __name__ == '__main__':
    main()
