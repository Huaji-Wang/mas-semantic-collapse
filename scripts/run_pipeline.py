"""Run measurement, ten motion weights, and topic comparisons (default stages 2-3)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commands(args):
    if int(args.start) > int(args.stop):
        raise ValueError('--from must not exceed --to')
    if args.motion_method == 'discrete' and (args.source == 'scale-replay' or args.scale_from or args.scales_file):
        raise ValueError('Replay comparisons and reference scales require motion weights')
    def command(name):
        return [sys.executable, str(ROOT / 'scripts' / name), '--tag', args.tag]
    result = []
    if int(args.start) <= 1 <= int(args.stop):
        replay = args.source == 'scale-replay'
        cmd = command('extract_scale_order_params.py' if replay else 'extract_order_params.py')
        cmd += ['--config', args.config, '--device', args.device, '--limit', str(args.limit if args.limit is not None else (0 if replay else 200)), '--cluster-distance', '0.55']
        if args.center_mean:
            cmd += ['--center-mean', args.center_mean]
        if replay:
            for flag, value in [('--replay', args.replay), ('--sims-dir', args.sims_dir)]:
                if value:
                    cmd += [flag, value]
            if args.skip_invalid:
                cmd += ['--skip-invalid']
        if args.allow_laptop_cuda:
            cmd += ['--allow-laptop-cuda']
        result.append(cmd)
    weighted = args.motion_method == 'weights'
    if int(args.start) <= 2 <= int(args.stop):
        cmd = command('classify_motion_weights.py' if weighted else 'classify_thread_motions.py')
        if weighted:
            cmd += ['--temperature', str(args.temperature)]
            for flag, value in [('--scale-from', args.scale_from), ('--scales-file', args.scales_file)]:
                if value:
                    cmd += [flag, value]
        result.append(cmd)
    if int(args.start) <= 3 <= int(args.stop):
        cmd = command('cluster_motion_weights.py' if weighted else 'cluster_titles.py')
        cmd += ['--config', args.config, '--cluster-method', args.cluster_method, '--min-cluster-size', str(args.min_cluster_size)]
        if args.title_backend:
            cmd += ['--title-backend', args.title_backend]
        if args.allow_laptop_cuda:
            cmd += ['--allow-laptop-cuda']
        result.append(cmd)
    return result


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from', dest='start', choices=('1', '2', '3'), default='2')
    ap.add_argument('--to', dest='stop', choices=('1', '2', '3'), default='3')
    ap.add_argument('--source', choices=('corpus', 'scale-replay'), default='corpus')
    ap.add_argument('--motion-method', choices=('weights', 'discrete'), default='weights')
    ap.add_argument('--tag', default='chik200')
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--title-backend')
    ap.add_argument('--cluster-method', choices=('kmeans', 'average_linkage'), default='kmeans')
    ap.add_argument('--min-cluster-size', type=int, default=15)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--allow-laptop-cuda', action='store_true')
    ap.add_argument('--replay')
    ap.add_argument('--sims-dir')
    ap.add_argument('--skip-invalid', action='store_true')
    ap.add_argument('--center-mean')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--scale-from')
    group.add_argument('--scales-file')
    ap.add_argument('--temperature', type=float, default=.25)
    return ap


def main():
    args = parser().parse_args()
    for cmd in commands(args):
        print('+', ' '.join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT), env={**os.environ, 'PYTHONUTF8': '1'})


if __name__ == '__main__':
    main()
