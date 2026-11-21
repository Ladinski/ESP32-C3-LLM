"""Train multiple examples; report independent held-out sequence metrics.

The model is deliberately small. No accuracy target is guaranteed, and
held-out scores are not proof of meaningful general language understanding.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from phase12_common import CHARS, TRAIN, HELDOUT, VOCAB, DIM, FF, MAX_TOKENS, pack4, fnv32, unpack_model, prompt_decode

torch.set_num_threads(1)


def fake_quant(x, scale, lo, hi):
    q = (x * scale).round().clamp(lo, hi) / scale
    return x + (q - x).detach()


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(VOCAB, DIM) * .22)
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(*shape) * .27)
            for shape in [(DIM, DIM)] * 4 + [(FF, DIM), (DIM, FF), (VOCAB, DIM)]
        ])
        offsets = torch.arange(DIM).unsqueeze(0) % 4
        positions = torch.arange(MAX_TOKENS).unsqueeze(1) % 4
        self.register_buffer('pos', (offsets == positions).float() * .125)

    def forward(self, tokens):
        n = len(tokens)
        emb = fake_quant(self.embed, 8., -128, 127)
        wq, wk, wv, wo, w1, w2, head = [fake_quant(w, 16., -8, 7) for w in self.w]
        x = emb[tokens] + self.pos[:n]
        z = F.layer_norm(x, (DIM,), eps=1e-5)
        q, k, v = (F.linear(z, w) for w in (wq, wk, wv))
        scores = (q @ k.T) / DIM ** .5
        scores = scores.masked_fill(~torch.ones(n, n, device=x.device, dtype=torch.bool).tril(), -1e9)
        ctx = scores.softmax(-1) @ v
        residual = x + F.linear(ctx, wo)
        ff = F.linear(F.relu(F.linear(F.layer_norm(residual, (DIM,), eps=1e-5), w1)), w2)
        return F.linear(F.layer_norm(residual + ff, (DIM,), eps=1e-5), head)


def tokens(s):
    return torch.tensor([CHARS.index(ch) for ch in s], dtype=torch.long)


def evaluate(model, corpus):
    loss_sum = 0.; n = 0; correct = 0
    with torch.no_grad():
        for s in corpus:
            ids = tokens(s)
            logits = model(ids[:-1])
            target = ids[1:]
            loss_sum += float(F.cross_entropy(logits, target, reduction='sum'))
            correct += int((logits.argmax(-1) == target).sum())
            n += len(target)
    return {'accuracy': round(correct/n, 5), 'correct': correct, 'characters': n,
            'cross_entropy': round(loss_sum/n, 5)}


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('--steps', type=int, default=1200)
    argp.add_argument('--seed', type=int, default=20261113)
    args = argp.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = Decoder()
    optim = torch.optim.Adam(model.parameters(), lr=.016)
    data = [tokens(s) for s in TRAIN]
    print('PHASE 12 | Multi-sequence INT4-aware character-transformer training')
    print(f'Vocabulary {VOCAB} | context {MAX_TOKENS} | train lines {len(TRAIN)} | held-out lines {len(HELDOUT)}')
    for step in range(1, args.steps + 1):
        optim.zero_grad()
        # Deterministic cycling across independently reset sequences
        batch = [data[(step * 3 + i) % len(data)] for i in range(4)]
        loss = sum(F.cross_entropy(model(seq[:-1]), seq[1:]) for seq in batch)/len(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optim.step()
        if step == 1 or step % 200 == 0:
            print(f'step {step}/{args.steps} | batch cross-entropy {loss.item():.4f}')
    model.eval()
    train_metrics = evaluate(model, TRAIN)
    heldout_metrics = evaluate(model, HELDOUT)
    with torch.no_grad():
        emb = (model.embed * 8).round().clamp(-128, 127).to(torch.int8).cpu().numpy()
        ws = [(w * 16).round().clamp(-8, 7).to(torch.int8).cpu().numpy() for w in model.w]
    payload = emb.tobytes() + b''.join(pack4(w) for w in ws)
    assert len(payload) == 1792
    em, weights = unpack_model(payload)
    prompt = [CHARS.index(x) for x in 'hi ']
    _, generated, scores = prompt_decode(em, weights, prompt, 12)
    sample = 'hi ' + ''.join(CHARS[t] for t in generated)
    output = Path(__file__).resolve().parent
    (output / 'phase12_model.bin').write_bytes(payload)
    metrics = dict(train=train_metrics, heldout=heldout_metrics, train_examples=TRAIN,
                   heldout_examples=HELDOUT, sample=sample, model_bytes=len(payload),
                   hash=f'{fnv32(payload):08x}', steps=args.steps, seed=args.seed,
                   warning='small, structurally similar held-out dataset; not proof of general language ability')
    (output / 'phase12_metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Train:', train_metrics)
    print('Held-out:', heldout_metrics)
    print('Sample:', repr(sample))
    print(f'SAVED phase12_model.bin ({len(payload)} bytes; FNV32 {fnv32(payload):08x})')


if __name__ == '__main__':
    main()
