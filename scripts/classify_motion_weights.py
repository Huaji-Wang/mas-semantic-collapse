"""Stage 2: ten continuous motion weights for arbitrary saved trajectories.

Uses the exact motion-mixture-v1 rules of the 311-pair research report.
No comments, embeddings, topic labels, or hard assignments are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from score_motion_mixtures import BOUNDS, MOTIONS, ROOT, TEMPERATURE, _scales, features, score


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as exc: raise ValueError(f'{path}:{number}: invalid JSON') from exc


def validate_rows(rows):
    if not rows: raise ValueError('No trajectories')
    seen=set(); sides={}; settings=set()
    for r in rows:
        tid=r.get('thread_id') or r.get('post_id')
        side=r.get('side','human')
        if not tid or side not in ('human','sim'): raise ValueError('Each row needs a thread/post ID and human/sim side')
        if (tid,side) in seen: raise ValueError(f'Duplicate trajectory: {tid}/{side}')
        seen.add((tid,side));sides.setdefault(tid,{})[side]=r
        n=r.get('n_windows')
        if not isinstance(n,int) or n<2: raise ValueError(f'{tid}/{side}: at least two windows required')
        if not isinstance(r.get('n_comments'),int) or r['n_comments']<n:
            raise ValueError(f'{tid}/{side}: invalid comment count')
        if 's_end' in r and not np.isfinite(r['s_end']):
            raise ValueError(f'{tid}/{side}: nonfinite final similarity')
        for key in ('sims_to_first','sigma','eff_dim','k_modes','chi','chi_step'):
            values=r.get(key,[])
            if len(values)!=n or not np.isfinite(np.asarray(values,dtype=float)).all():
                raise ValueError(f'{tid}/{side}: invalid {key} series')
        settings.add(tuple(r.get(k) for k in ('embedder','center_mean','window_size','support_size','k_cluster_distance')))
    if len(settings)>1: raise ValueError('Mixed embedding/centering/window settings; score compatible trajectories together')
    if any(side=='sim' for _,side in seen):
        for tid,pair in sides.items():
            if set(pair)!= {'human','sim'}: raise ValueError(f'Unpaired thread: {tid}')
            for k in ('n_windows','n_comments'):
                if pair['human'].get(k)!=pair['sim'].get(k): raise ValueError(f'{tid}: paired {k} mismatch')


def aggregate_weights(rows):
    result={'n_trajectories':len(rows),'motion_order':MOTIONS,'by_side':{}}
    for side in ('human','sim'):
        block=[r for r in rows if r['side']==side]
        if not block:continue
        a=np.array([[r['weights'][m] for m in MOTIONS] for r in block])
        result['by_side'][side]={'n':len(block),'mean_weights':dict(zip(MOTIONS,a.mean(0).tolist())),
                                 'thread_equivalent_mass':dict(zip(MOTIONS,a.sum(0).tolist()))}
    pairs={}
    for r in rows:pairs.setdefault(r['thread_id'],{})[r['side']]=r
    deltas=[np.array([p['sim']['weights'][m]-p['human']['weights'][m] for m in MOTIONS]) for p in pairs.values() if set(p)=={'human','sim'}]
    if deltas:
        result['paired']={'n':len(deltas),'mean_sim_minus_human_pp':dict(zip(MOTIONS,(np.mean(deltas,axis=0)*100).tolist())),
                          'mean_total_variation':float(np.mean([np.abs(d).sum()/2 for d in deltas]))}
    return result


def classify_rows(rows,scales=None,temperature=TEMPERATURE):
    validate_rows(rows)
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('temperature must be finite and positive')
    if scales is None:
        human=[features(r) for r in rows if r.get('side','human')=='human']
        if not human:raise ValueError('Human trajectories are required to fit scales')
        scales=_scales(human)
    required={'sigma','dim','k','chi','m','step','chi_std'}
    if not required.issubset(scales) or any(not np.isfinite(scales[k]) or scales[k]<=0 for k in required):
        raise ValueError('Scales must contain seven finite positive feature scales')
    out=[]
    scale_fingerprint=hashlib.sha256(json.dumps(scales,sort_keys=True).encode()).hexdigest()
    for r in rows:
        s=score(features(r),scales,temperature)
        item={k:r.get(k) for k in ('post_id','title','subreddit','n_comments','n_windows')}
        item.update({'thread_id':r.get('thread_id') or r['post_id'],'side':r.get('side','human'),
                     'method':'motion-mixture-v1','temperature':temperature,
                     'scale_fingerprint':scale_fingerprint,
                     'raw_scores':dict(zip(MOTIONS,s['raw_scores'])),
                     'compatibility':dict(zip(MOTIONS,s['compatibility'])),
                     'weights':dict(zip(MOTIONS,s['weights'])),
                     'dominant_motion':s['dominant_motion'],'entropy_normalized':s['entropy_normalized'],
                     'shape_features_resolved':r['n_windows']>=4,
                     'interpretation':'Normalized heuristic compatibility; not calibrated probabilities.'})
        out.append(item)
    return out,scales


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag',default='chik200')
    ap.add_argument('--input',type=Path,nargs='+',help='One or more compatible order-parameter JSONL files')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--summary',type=Path)
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--scale-from',type=Path,help='Fit scales on the human rows of this reference JSONL')
    group.add_argument('--scales-file',type=Path,help='Frozen scales JSON or previous summary containing scales')
    ap.add_argument('--temperature',type=float,default=TEMPERATURE)
    args=ap.parse_args()
    paths=args.input or [ROOT/f'outputs/order_params/order_params_{args.tag}.jsonl']
    rows=[r for p in paths for r in read_rows(p)]
    validate_rows(rows)
    scales=None;scale_source='human rows of input trajectories'
    if args.scales_file:
        saved=json.loads(args.scales_file.read_text(encoding='utf-8'));scales=saved.get('scales',saved)
        if 'measurement_protocol' in saved:
            if saved['measurement_protocol']!={k:rows[0].get(k) for k in saved['measurement_protocol']}:
                raise ValueError('Frozen scales and input measurement settings differ')
        scale_source=str(args.scales_file)
    if args.scale_from:
        reference=list(read_rows(args.scale_from));validate_rows(reference)
        keys=('embedder','center_mean','window_size','support_size','k_cluster_distance')
        if tuple(reference[0].get(k) for k in keys)!=tuple(rows[0].get(k) for k in keys):
            raise ValueError('Reference and input measurement settings differ')
        scales=_scales([features(r) for r in reference if r.get('side','human')=='human'])
        scale_source=str(args.scale_from)
    labels,scales=classify_rows(rows,scales,args.temperature)
    out=args.output or ROOT/f'outputs/labels/thread_motion_weights_{args.tag}.jsonl'
    summary=args.summary or ROOT/f'outputs/metrics/thread_motion_weights_{args.tag}.json'
    payload={**aggregate_weights(labels),'method':'motion-mixture-v1','temperature':args.temperature,
             'score_bounds':dict(zip(MOTIONS,BOUNDS)),'scales':scales,'scale_source':scale_source,
             'measurement_protocol':{k:rows[0].get(k) for k in ('embedder','center_mean','window_size','support_size','k_cluster_distance')},
             'short_trajectories':sum(not r['shape_features_resolved'] for r in labels),
             'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
             'interpretation':'Mean weights and fractional thread-equivalent mass; not hard-label counts.'}
    for p in [args.scale_from,args.scales_file]:
        if p:payload['source_hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [out,summary]:p.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(out.suffix+'.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in labels),encoding='utf-8');tmp.replace(out)
    summary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'trajectories':len(labels),'pairs':payload.get('paired',{}).get('n',0),'weights':str(out),'summary':str(summary)}))


if __name__=='__main__':main()
