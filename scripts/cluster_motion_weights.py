"""Stage 3: shared title clusters and continuous human/simulation comparisons."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from classify_motion_weights import MOTIONS, ROOT, aggregate_weights, read_rows
from cluster_titles import _choose_k, _is_unusable, _keywords, _title_cfg
from mas_collapse.config import load_config
from mas_collapse.embed.backend import build_embedder


def shared_threads(rows):
    """Cluster each original title once, keeping paired sides in the same topic."""
    threads, seen, protocols = {}, set(), set()
    for row in rows:
        tid, side = row['thread_id'], row['side']
        if side not in ('human', 'sim') or (tid, side) in seen:
            raise ValueError('Invalid or duplicate trajectory')
        seen.add((tid, side))
        protocols.add((row['method'], row['temperature'], row.get('scale_fingerprint')))
        if set(row['weights']) != set(MOTIONS):
            raise ValueError('Expected all ten named motion weights')
        w = np.array([row['weights'][m] for m in MOTIONS])
        if not np.isfinite(w).all() or (w < 0).any() or abs(w.sum() - 1) > 1e-9:
            raise ValueError('Invalid motion weight vector')
        title = row.get('title') or ''
        if tid in threads and threads[tid] != title:
            raise ValueError(f'{tid}: paired titles differ')
        threads[tid] = title
    if not rows or len(protocols) != 1 or next(iter(protocols))[0] != 'motion-mixture-v1':
        raise ValueError('Expected one motion-mixture-v1 protocol')
    if any(side == 'sim' for _, side in seen):
        if any((tid, side) not in seen for tid in threads for side in ('human', 'sim')):
            raise ValueError('Human/simulation comparisons require complete pairs')
    return threads


def render(payload):
    sections = []
    for topic in payload['topics']:
        table = []
        for motion in MOTIONS:
            cells = []
            for side in ('human', 'sim'):
                block = topic['by_side'].get(side)
                cells.append(f"{100 * block['mean_weights'][motion]:.2f}%" if block else '—')
            delta = topic.get('paired', {}).get('mean_sim_minus_human_pp', {}).get(motion)
            cells.append(f'{delta:+.2f}' if delta is not None else '—')
            table.append('<tr><th>' + motion + '</th>' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>')
        sections.append(f"<h2>{html.escape(str(topic['cluster_id']))}: {html.escape(', '.join(topic['keywords']))}</h2>"
                        f"<p>{topic['n_threads']} original threads</p><table><tr><th>Motion</th><th>Human mean</th>"
                        '<th>Simulation mean</th><th>Paired difference (pp)</th></tr>' + ''.join(table) + '</table>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Topic and motion weights</title>'
            '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px}table{border-collapse:collapse;width:100%}'
            'td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style><h1>Motion mixtures by topic</h1>'
            '<p>Topics are exploratory clusters of original titles. Both sides share the same topic assignment. '
            'Weights describe heuristic compatibility, not calibrated probabilities. Differences are descriptive; '
            'no significance test is implied.</p>' + ''.join(sections) + '</html>')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='chik200')
    ap.add_argument('--input', type=Path)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--title-backend', default=None)
    ap.add_argument('--allow-laptop-cuda', action='store_true')
    ap.add_argument('--cluster-method', choices=('kmeans', 'average_linkage'), default='kmeans')
    ap.add_argument('--min-cluster-size', type=int, default=15)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=12)
    args = ap.parse_args()
    if args.k_min < 2 or args.k_max < args.k_min or args.min_cluster_size < 1:
        raise ValueError('Invalid cluster settings')
    source = args.input or ROOT / f'outputs/labels/thread_motion_weights_{args.tag}.jsonl'
    rows = list(read_rows(source))
    threads = shared_threads(rows)
    usable = [tid for tid, title in threads.items() if not _is_unusable(title)]
    titles = [threads[tid] for tid in usable]
    assignments = {tid: 'unusable' for tid in threads}
    search, keywords = [], {}
    title_cfg = _title_cfg(load_config(args.config), args.title_backend, args.allow_laptop_cuda)
    if titles:
        if len(titles) < 3 or len(set(titles)) == 1:
            labels = np.ones(len(titles), dtype=int)
        else:
            vecs = np.asarray(build_embedder(title_cfg).embed(titles), dtype=float)
            vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
            if not np.isfinite(vecs).all():
                raise ValueError('Nonfinite title embeddings')
            if np.allclose(vecs, vecs[0]):
                labels = np.ones(len(titles), dtype=int)
            else:
                hi = min(args.k_max, len(titles) - 1)
                _, labels, search = _choose_k(vecs, min(args.k_min, hi), hi, args.cluster_method, args.min_cluster_size)
        keywords = _keywords(titles, labels)
        assignments.update({tid: int(label) for tid, label in zip(usable, labels)})
    topics = []
    for cluster in sorted(set(assignments.values()), key=str):
        block = [r for r in rows if assignments[r['thread_id']] == cluster]
        topics.append({'cluster_id': cluster, 'n_threads': len({r['thread_id'] for r in block}),
                       'keywords': keywords.get(cluster, []), **aggregate_weights(block)})
    payload = {'method': 'motion-mixture-v1', 'topic_method': args.cluster_method,
               'title_embedding': {k: title_cfg['embedding'].get(k) for k in ('backend', 'model_name', 'openai_model', 'hash_dim', 'device')}, 'cluster_search': search,
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'overall': aggregate_weights(rows), 'topics': topics,
               'interpretation': 'Exploratory title topics; paired differences in percentage points. Fractional mass is not a count of people.'}
    folder = ROOT / 'outputs/title_clusters'; folder.mkdir(parents=True, exist_ok=True)
    out = folder / f'motion_weights_{args.tag}.json'
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (folder / f'thread_topics_{args.tag}.jsonl').write_text(''.join(json.dumps({'thread_id': tid, 'cluster_id': cid}) + '\n' for tid, cid in assignments.items()), encoding='utf-8')
    page = folder / f'motion_weights_{args.tag}.html'
    page.write_text(render(payload), encoding='utf-8')
    print(json.dumps({'threads': len(threads), 'topics': len(topics), 'summary': str(out), 'report': str(page)}))


if __name__ == '__main__':
    main()
