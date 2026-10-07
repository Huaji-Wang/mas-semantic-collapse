"""Versioned, ten-component descriptive motion mixtures from saved trajectories.

No generation, embedding, or modification of the legacy nine-class outputs.
Weights are normalized heuristic compatibility, not calibrated probabilities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from classify_thread_motions import features, score_row, _scales, _stable

ROOT = Path(__file__).resolve().parents[1]
TAGS = ['persona_148', 'persona_lenfree']
MOTIONS = ['translation','diffusion','condensation','fission','crystallization',
           'vortex','orbit','breathing','cascade','filamentation']
ZH = ['平移','扩散','凝聚','裂变','结晶','涡流','轨道','呼吸','级联','成丝']
# Conservative algebraic bounds on the existing nine additive rules; the
# tenth rule below is bounded in [0,3]. These are not data-fitted extrema.
BOUNDS = [(-.8,3.6),(0,2.9),(0,2.6),(0,3.5),(-.6,2.9),(-1.2,2.7),
          (0,2.9),(0,2.7),(0,2.7),(0,3)]
TEMPERATURE = .25


def read(rel):
    return [json.loads(x) for x in (ROOT/rel).read_text(encoding='utf-8').splitlines() if x.strip()]


def score(f,sc,temperature=TEMPERATURE,filament_scale=1):
    raw = score_row(f,sc)
    # One-sided dimension-decline evidence: no decline gives zero evidence.
    decline = -np.expm1(-max(0.,-f['d_dim'])/(sc['dim']*filament_scale))
    raw['filamentation'] = float(decline*(2+.5*_stable(f['d_sigma'],sc['sigma'])+.5*_stable(f['d_k'],sc['k'])))
    scores = np.array([raw[m] for m in MOTIONS])
    lo,hi = np.array(BOUNDS).T
    assert np.isfinite(scores).all()
    assert np.all(scores>=lo-1e-9) and np.all(scores<=hi+1e-9)
    compatibility = (scores-lo)/(hi-lo)
    exponents = (compatibility-compatibility.max())/temperature
    weights = np.exp(exponents); weights /= weights.sum()
    assert np.isfinite(weights).all() and (weights>=0).all()
    assert abs(weights.sum()-1)<1e-12
    return {'raw_scores':scores.tolist(),'compatibility':compatibility.tolist(),'weights':weights.tolist(),
            'entropy_normalized':float(-sum(weights*np.log(weights))/np.log(10)),
            'dominant_motion':MOTIONS[int(np.argmax(weights))], 'dimension_decline_evidence':float(decline)}


def summarize(rows):
    n=len(rows)
    h=np.array([r['human']['weights'] for r in rows]); s=np.array([r['sim']['weights'] for r in rows])
    if not n:return {'n':0,'human_mean':None,'sim_mean':None,'delta_pp':None,'mean_total_variation':None}
    return {'n':n,'human_mean':h.mean(0).tolist(),'sim_mean':s.mean(0).tolist(),
            'delta_pp':((s-h).mean(0)*100).tolist(),
            'mean_total_variation':float(np.abs(s-h).sum(1).mean()/2),
            'positive_pairs':((s-h)>0).sum(0).tolist()}


def main():
    trajectory={}; hashes={}
    for tag in TAGS:
        rel=f'outputs/order_params/order_params_{tag}_centered.jsonl'
        hashes[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
        for row in read(rel):
            key=(tag,row['thread_id'],row['side'])
            assert key not in trajectory
            trajectory[key]=row
    sc=_scales([features(r) for (tag,tid,side),r in trajectory.items() if side=='human'])
    batch_sc={tag:_scales([features(r) for (b,tid,side),r in trajectory.items() if side=='human' and b==tag]) for tag in TAGS}
    annotations=read('outputs/labels/content_topics_persona_311.jsonl')
    book=json.loads((ROOT/'docs/content_topic_codebook.json').read_text(encoding='utf-8'))
    hashes.update(book['source_hashes'])
    for rel in ['outputs/labels/content_topics_persona_311.jsonl','reference/corpus_mean.npy',
                'scripts/classify_thread_motions.py','scripts/score_motion_mixtures.py',
                'mas_collapse/metrics/order_params.py']:
        hashes[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    rows=[]; variations={}
    settings=[('tau_015',.15,1,False),('tau_050',.5,1,False),
              ('filament_scale_05',.25,.5,False),('filament_scale_15',.25,1.5,False),
              ('batch_scales',.25,1,True)]
    for name,*_ in settings:variations[name]=[]
    for a in annotations:
        h=trajectory[(a['batch'],a['thread_id'],'human')]; s=trajectory[(a['batch'],a['thread_id'],'sim')]
        assert h['n_comments']==s['n_comments'] and h['n_windows']==s['n_windows']
        entry={k:a[k] for k in ['thread_id','batch','title','topic','topic_zh','topic_en','annotation_status']}
        entry.update({'n_comments':h['n_comments'],'n_windows':h['n_windows'],'short_trajectory':h['n_windows']<4})
        for side,r in [('human',h),('sim',s)]:
            f=features(r)
            entry[side]={'features':f,**score(f,sc),
                         'series':{k:r[k] for k in ['sims_to_first','sigma','eff_dim','k_modes','chi','chi_step','n_support']},
                         'late':{k:r[k] for k in ['s_end','sigma_late','eff_dim_late','k_modes_late','chi_late']}}
        rows.append(entry)
        for name,tau,fil,batch in settings:
            row={'topic':a['topic'],'batch':a['batch']}
            for side in ('human','sim'):
                row[side]=score(entry[side]['features'],batch_sc[a['batch']] if batch else sc,tau,fil)
            variations[name].append(row)
    assert len(rows)==311 and len(trajectory)==622
    topics=[]
    for t in book['topics']:
        selected=[r for r in rows if r['topic']==t['topic']]
        topics.append({**t,'pooled':summarize(selected),
                       'batches':{tag:summarize([r for r in selected if r['batch']==tag]) for tag in TAGS},
                       'long_only':summarize([r for r in selected if not r['short_trajectory']]),
                       'without_mixed':summarize([r for r in selected if r['annotation_status']!='mixed'])})
    topics.sort(key=lambda t:-t['pooled']['n'])
    sensitivity={name:{'overall':summarize(rs),'topics':{t['topic']:summarize([r for r in rs if r['topic']==t['topic']]) for t in topics}} for name,rs in variations.items()}
    sensitivity['long_only']={'overall':summarize([r for r in rows if not r['short_trajectory']]),'topics':{t['topic']:t['long_only'] for t in topics}}
    sensitivity['without_mixed']={'overall':summarize([r for r in rows if r['annotation_status']!='mixed']),'topics':{t['topic']:t['without_mixed'] for t in topics}}
    data={'version':'motion-mixture-v1','motion_order':MOTIONS,'motion_zh':ZH,'temperature':TEMPERATURE,
          'score_bounds':dict(zip(MOTIONS,BOUNDS)), 'scale_source':'MAD of 311 pooled human feature vectors; identical scales on both sides and batches',
          'scales':sc,'batch_scales_for_sensitivity':batch_sc,'source_hashes':hashes,
          'interpretation':'Normalized heuristic compatibility weights, not calibrated probabilities or a generative mixture model.',
          'short_n':sum(r['short_trajectory'] for r in rows),'totals':summarize(rows),
          'batches':{tag:summarize([r for r in rows if r['batch']==tag]) for tag in TAGS},
          'topics':topics,'sensitivity':sensitivity,'annotation':book,'rows':rows}
    for rel in ['docs/report_motion_mixtures.json','outputs/metrics/motion_mixtures_persona_311.json']:
        (ROOT/rel).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    out=ROOT/'outputs/labels/thread_motion_mixtures_persona_311.jsonl'
    out.write_text(''.join(json.dumps({k:v for k,v in row.items()},ensure_ascii=False,allow_nan=False)+'\n' for row in rows),encoding='utf-8')
    print(json.dumps({'n':311,'short_n':data['short_n'],'overall':data['totals'],
                      'topic_deltas':{t['zh']:dict(zip(MOTIONS,np.round(t['pooled']['delta_pp'],2))) for t in topics}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
