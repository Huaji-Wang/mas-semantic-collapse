"""Validate full-position replay pairs, then measure saved human/sim comments.

This command never generates comments. --audit-only does not load an embedder.
Per-thread checkpoints are reused only when source text and measurement settings
have matching fingerprints. Invalid or incomplete pairs fail closed by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from mas_collapse.config import load_config
from mas_collapse.embed.backend import build_embedder
from mas_collapse.metrics.order_params import extract_order_params, summarize_order_params
from scripts.classify_motion_weights import read_rows


def validate_pair(thread,path):
    tid=thread['post_id'];human=thread.get('turns',[])
    records=list(read_rows(path))
    if not records or records[0].get('type')!='meta':raise ValueError('missing metadata header')
    meta=records[0];sim=records[1:]
    if meta.get('post_id')!=tid or path.stem!=tid:raise ValueError('thread ID mismatch')
    if not human or meta.get('n_turns')!=len(human) or len(sim)!=len(human):raise ValueError('incomplete or mismatched turn count')
    for index,(h,s) in enumerate(zip(human,sim)):
        for k in ('position','author','comment_id','alias'):
            if k not in h or k not in s or h[k]!=s[k]:raise ValueError(f'turn {index}: {k} mismatch')
        if h['position']!=index:raise ValueError(f'turn {index}: non-contiguous position')
        if s.get('error') or not isinstance(s.get('body'),str) or not s['body'].strip():raise ValueError(f'turn {index}: failed/empty simulation')
        if not isinstance(h.get('body'),str) or not h['body'].strip():raise ValueError(f'turn {index}: empty human comment')
        shown=s.get('shown_indices')
        if not isinstance(shown,list) or len(set(shown))!=len(shown) or any(type(x)!=int or x<0 or x>=index for x in shown):
            raise ValueError(f'turn {index}: invalid visible-history positions')
    return human,sim,meta


def iter_pairs(replay,sims,limit=0):
    seen=set()
    for i,thread in enumerate(read_rows(replay)):
        if limit and i>=limit:break
        tid=thread.get('post_id','')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',tid) or tid in seen:raise ValueError('Invalid/duplicate replay thread ID')
        seen.add(tid)
        path=sims/f'{tid}.jsonl'
        try:
            h,s,m=validate_pair(thread,path)
            yield thread,h,s,m,None
        except (OSError,ValueError,KeyError,TypeError) as exc:
            yield thread,None,None,None,str(exc)


def _text_sha(texts):
    return [hashlib.sha256(t.encode('utf-8')).hexdigest()[:16] for t in texts]


def _seed_index(seed_dir):
    """comment_id -> npz path, over <post_id>.npz written by extract_order_params.py --save-embeddings."""
    index={}
    for f in sorted(Path(seed_dir).glob('*.npz')):
        z=np.load(f,allow_pickle=False)
        for cid in z['comment_ids'].tolist():index.setdefault(str(cid),f)
    return index


def _side_vectors(get_model,comments,side,tid,store,seed,seed_name,counts):
    """Raw (uncentred) vectors for one side of one thread.

    Order of preference: the thread's own stored file (only if every comment text
    is unchanged), then seed vectors by comment_id (human side only), then the
    embedder. Returns (vectors, embedder_name). With store=None and seed=None this
    is exactly model.embed(bodies)."""
    texts=[r['body'] for r in comments];sha=_text_sha(texts)
    path=store/f'{tid}_{side}.npz' if store is not None else None
    if path is not None and path.exists():
        z=np.load(path,allow_pickle=False)
        if z['text_sha'].tolist()==sha:
            counts['stored']+=len(texts);return z['embs'].astype(np.float32),str(z['embedder'])
    vecs=[None]*len(texts);name=None
    if seed is not None and side=='human':
        by_file={}
        for i,r in enumerate(comments):
            f=seed.get(str(r['comment_id']))
            if f is not None:by_file.setdefault(f,[]).append(i)
        for f,rows in by_file.items():
            z=np.load(f,allow_pickle=False)
            if str(z['embedder'])!=seed_name:raise ValueError(f'{f}: embedder {z["embedder"]} != {seed_name}')
            pos={str(c):j for j,c in enumerate(z['comment_ids'].tolist())};e=z['embs']
            for i in rows:vecs[i]=np.asarray(e[pos[str(comments[i]['comment_id'])]],dtype=np.float32)
        name=seed_name if by_file else None
        counts['seeded']+=sum(v is not None for v in vecs)
    missing=[i for i,v in enumerate(vecs) if v is None]
    if missing:
        model=get_model();fresh=np.asarray(model.embed([texts[i] for i in missing]),dtype=np.float32)
        for k,i in enumerate(missing):vecs[i]=fresh[k]
        if name is not None and name!=model.name:raise ValueError(f'seed embedder {name} != model {model.name}')
        name=model.name;counts['embedded']+=len(missing)
    e=np.stack(vecs).astype(np.float32)
    if path is not None:
        part=path.with_name(path.name+'.tmp')
        with open(part,'wb') as fh:
            np.savez(fh,embs=e,comment_ids=np.asarray([str(r['comment_id']) for r in comments]),
                     text_sha=np.asarray(sha),embedder=np.asarray(name))
        part.replace(path)
    return e,name


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay',type=Path,default=ROOT/'outputs/scale5000/replay.jsonl')
    ap.add_argument('--sims-dir',type=Path,default=ROOT/'outputs/sims/scale5000/low_deepseek')
    ap.add_argument('--tag',default='scale5000_low_deepseek_centered')
    ap.add_argument('--config',default='configs/default.yaml')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--skip-invalid',action='store_true',help='Explicitly exclude invalid pairs; list them in the audit')
    ap.add_argument('--limit',type=int,default=0,help='Replay-post limit, 0 means all')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--allow-laptop-cuda',action='store_true')
    ap.add_argument('--backend',default=None)
    ap.add_argument('--center-mean',type=Path,default=ROOT/'reference/corpus_mean.npy')
    ap.add_argument('--uncentered',action='store_true',help='Explicit uncentered analysis; required for hashing smoke tests')
    ap.add_argument('--window-size',type=int,default=10)
    ap.add_argument('--support-size',type=int,default=30)
    ap.add_argument('--cluster-distance',type=float,default=.55)
    ap.add_argument('--vector-store',type=Path,default=None,
                    help='[patch] persist raw per-thread, per-side comment vectors here and reuse them on later runs '
                         'when every comment text is unchanged (re-measuring then needs no embedder)')
    ap.add_argument('--seed-vectors',type=Path,default=None,
                    help='[patch] dir of <post_id>.npz from extract_order_params.py --save-embeddings; human comments '
                         'found there by comment_id reuse those vectors (same embedder required)')
    args=ap.parse_args()
    if args.limit<0 or args.window_size<1 or args.support_size<args.window_size:raise ValueError('Invalid limit/window/support size')
    audit={'valid_pairs':0,'comments_per_side':0,'truncated_sim_comments':0,'excluded':[],
           'replay':str(args.replay),'sims_dir':str(args.sims_dir),'limit':args.limit}
    for t,h,s,m,error in iter_pairs(args.replay,args.sims_dir,args.limit):
        if error or len(h or [])<=args.window_size:
            audit['excluded'].append({'thread_id':t['post_id'],'reason':error or 'fewer than two windows'})
        else:
            audit['valid_pairs']+=1;audit['comments_per_side']+=len(h)
            audit['truncated_sim_comments']+=sum(bool(r.get('truncated')) for r in s)
    audit_path=ROOT/f'outputs/metrics/replay_audit_{args.tag}.json';audit_path.parent.mkdir(parents=True,exist_ok=True)
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({**{k:v for k,v in audit.items() if k!='excluded'},'excluded_pairs':len(audit['excluded']),'audit':str(audit_path)}),flush=True)
    if args.audit_only:return
    if audit['excluded'] and not args.skip_invalid:raise SystemExit('Invalid pairs found; inspect audit or explicitly use --skip-invalid')
    if not audit['valid_pairs']:raise SystemExit('No valid pairs')
    if args.device.lower().startswith('cuda') and not args.allow_laptop_cuda:raise SystemExit('Use CPU on this machine; CUDA requires --allow-laptop-cuda')
    cfg=load_config(args.config);cfg['embedding']['device']=args.device
    if args.backend:cfg['embedding']['backend']=args.backend
    mu=None if args.uncentered else np.load(args.center_mean,allow_pickle=False).astype(np.float32)
    if mu is not None and (mu.ndim!=1 or not np.isfinite(mu).all()):raise ValueError('Invalid reference mean')
    settings={k:cfg['embedding'].get(k) for k in ('backend','model_name','openai_model','hash_dim','device','batch_size')}
    settings.update({'window_size':args.window_size,'support_size':args.support_size,'cluster_distance':args.cluster_distance,
                     'center_sha256':None if mu is None else hashlib.sha256(args.center_mean.read_bytes()).hexdigest(),
                     'measurement_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in [Path(__file__), ROOT/'mas_collapse/metrics/order_params.py', ROOT/'mas_collapse/metrics/semantic.py', ROOT/'mas_collapse/embed/backend.py'])).hexdigest()})
    out=args.output or ROOT/f'outputs/order_params/order_params_{args.tag}.jsonl'
    out.parent.mkdir(parents=True,exist_ok=True);cache=out.parent/(out.stem+'_cache');cache.mkdir(exist_ok=True)
    model=None;written=reused=0;tmp=out.with_suffix(out.suffix+'.tmp')
    def get_model():
        nonlocal model
        if model is None:model=build_embedder(cfg)
        return model
    store=args.vector_store
    if store is not None:store.mkdir(parents=True,exist_ok=True)
    seed=seed_name=None
    if args.seed_vectors is not None:
        if cfg['embedding'].get('backend','bge_m3') not in ('bge_m3','bge-m3'):raise SystemExit('--seed-vectors requires the bge_m3 backend')
        seed_name=f"bge_m3:{cfg['embedding'].get('model_name','BAAI/bge-m3')}"
        seed=_seed_index(args.seed_vectors);print(f'[patch] seed index: {len(seed):,} comment vectors from {args.seed_vectors}',flush=True)
    counts={'stored':0,'seeded':0,'embedded':0}
    with tmp.open('w',encoding='utf-8') as stream:
        for t,h,s,m,error in iter_pairs(args.replay,args.sims_dir,args.limit):
            if error or len(h or [])<=args.window_size:continue
            fingerprint=hashlib.sha256(json.dumps({'settings':settings,'thread':t,'sim':s},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            checkpoint=cache/f'{t["post_id"]}.json';stored=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {}
            if stored.get('fingerprint')==fingerprint:rows=stored['rows'];reused+=1
            else:
                rows=[]
                for side,comments in [('human',h),('sim',s)]:
                    e,emb_name=_side_vectors(get_model,comments,side,t['post_id'],store,seed,seed_name,counts)
                    if mu is not None:
                        if e.shape[1]!=len(mu):raise ValueError('Reference-mean dimension does not match embedder')
                        e=e-mu;e=e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
                    trajectory=extract_order_params(e,window_size=args.window_size,support_size=args.support_size,cluster_distance=args.cluster_distance)
                    rows.append({'post_id':f'{t["post_id"]}_{side}','thread_id':t['post_id'],'side':side,
                                 'title':t.get('title',''),'subreddit':t.get('subreddit'),'n_comments':len(comments),
                                 'embedder':emb_name,'center_mean':None if mu is None else settings['center_sha256'],
                                 'measurement_settings':settings,'source_fingerprint':fingerprint,
                                 **trajectory,**summarize_order_params(trajectory,late_windows=3)})
                part=checkpoint.with_suffix('.json.tmp');part.write_text(json.dumps({'fingerprint':fingerprint,'rows':rows},ensure_ascii=False)+'\n',encoding='utf-8');part.replace(checkpoint)
            for r in rows:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
            stream.flush();written+=1
            if written%25==0:print(f'measured pairs={written}; reused={reused}; vectors={counts}',flush=True)
    tmp.replace(out)
    print(json.dumps({'paired_threads':written,'reused_pairs':reused,'trajectories':written*2,'vectors':counts,
                      'model_loaded':model is not None,'output':str(out)}))


if __name__=='__main__':main()
