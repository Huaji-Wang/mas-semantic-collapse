"""Portable motion-mixture and replay regression tests; no models or corpus needed."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from classify_motion_weights import MOTIONS, aggregate_weights, classify_rows
from cluster_motion_weights import shared_threads
from extract_scale_order_params import validate_pair
from run_pipeline import commands, parser
from score_motion_mixtures import score
from mas_collapse.embed.backend import HashingEmbedder


def trajectory(tid='a', side='human', shift=0):
    return {'thread_id': tid, 'post_id': tid + '_' + side, 'side': side, 'title': 'A shared title',
            'n_comments': 40, 'n_windows': 4, 'embedder': 'fixture',
            'sims_to_first': [1., .9, .7, .8], 's_end': .8,
            'sigma': [.2, .3, .4 + shift, .3 + shift], 'eff_dim': [5., 4., 3., 2.],
            'k_modes': [1, 1, 2, 2], 'chi': [0., .3, .5, .2], 'chi_step': [0., .3, .2, .3]}


class MotionTests(unittest.TestCase):
    def test_ten_weights_conserve_mass(self):
        labels, _ = classify_rows([trajectory()])
        self.assertEqual(set(labels[0]['weights']), set(MOTIONS))
        self.assertAlmostEqual(sum(labels[0]['weights'].values()), 1)
        self.assertTrue(all(w >= 0 for w in labels[0]['weights'].values()))

    def test_scales_ignore_simulation(self):
        human = [trajectory(str(i), shift=i * .1) for i in range(6)]
        sim = [trajectory(str(i), 'sim', 20 + i) for i in range(6)]
        h, scales = classify_rows(human)
        both, paired_scales = classify_rows(human + sim)
        self.assertEqual(scales, paired_scales)
        self.assertEqual(h, both[:6])

    def test_malformed_and_unpaired_rejected(self):
        for rows in ([], [trajectory(), trajectory()], [trajectory(side='sim')],
                     [trajectory(), trajectory('b', 'sim')]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                classify_rows(rows)
        for key, value in [('sigma', [float('nan')] * 4), ('chi', [0]), ('n_windows', 1)]:
            row = trajectory(); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                classify_rows([row])

    def test_invalid_temperature_and_scale(self):
        for temperature in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                classify_rows([trajectory()], temperature=temperature)
        with self.assertRaises(ValueError):
            classify_rows([trajectory()], scales={'sigma': 0})

    def test_short_shape_evidence_flag(self):
        r = trajectory(); r['n_windows'] = 2; r['n_comments'] = 20
        for key in ('sims_to_first', 'sigma', 'eff_dim', 'k_modes', 'chi', 'chi_step'):
            r[key] = r[key][:2]
        labels, _ = classify_rows([r])
        self.assertFalse(labels[0]['shape_features_resolved'])

    def test_fractional_mass_and_paired_difference(self):
        labels, _ = classify_rows([trajectory(), trajectory(side='sim', shift=.8)])
        summary = aggregate_weights(labels)
        for side in ('human', 'sim'):
            self.assertAlmostEqual(sum(summary['by_side'][side]['thread_equivalent_mass'].values()), 1)
        expected = 100 * (labels[1]['weights']['diffusion'] - labels[0]['weights']['diffusion'])
        self.assertAlmostEqual(summary['paired']['mean_sim_minus_human_pp']['diffusion'], expected)
        self.assertEqual(summary['paired']['n'], 1)

    def test_shared_topic_identity_and_bad_vector(self):
        rows, _ = classify_rows([trajectory(), trajectory(side='sim')])
        self.assertEqual(shared_threads(rows), {'a': 'A shared title'})
        bad = copy.deepcopy(rows); bad[1]['title'] = 'different'
        with self.assertRaises(ValueError): shared_threads(bad)
        bad = copy.deepcopy(rows); bad[1]['weights']['vortex'] += .1
        with self.assertRaises(ValueError): shared_threads(bad)

    def test_frozen_reference_vector(self):
        # All stable, stationary descriptors: independent closed-form baseline.
        f = dict(d_sigma=0., d_dim=0., d_k=0., d_chi=0., m_leave=0.,
                 chi_step_mean=0., chi_late_std=0., m_wave=0., chi_wave=0.,
                 sigma_wave=0., m_step=0., chi_step_shape=0.)
        scales = dict.fromkeys(('sigma', 'dim', 'k', 'chi', 'm', 'step', 'chi_std'), 1.)
        result = score(f, scales)
        raw = [2.2, 1.8, 1.5, 1.9, 2.9, .8, .3, .3, .3, 0.]
        np.testing.assert_allclose(result['raw_scores'], raw, atol=1e-14)
        lo = np.array([-.8, 0, 0, 0, -.6, -1.2, 0, 0, 0, 0])
        hi = np.array([3.6, 2.9, 2.6, 3.5, 2.9, 2.7, 2.9, 2.7, 2.7, 3])
        exp = np.exp(((np.array(raw) - lo) / (hi - lo)) / .25)
        np.testing.assert_allclose(result['weights'], exp / exp.sum(), atol=1e-14)

    def test_default_and_legacy_routes(self):
        weighted = commands(parser().parse_args([]))
        self.assertEqual([Path(c[1]).name for c in weighted], ['classify_motion_weights.py', 'cluster_motion_weights.py'])
        legacy = commands(parser().parse_args(['--motion-method', 'discrete']))
        self.assertEqual(Path(legacy[0][1]).name, 'classify_thread_motions.py')
        replay = commands(parser().parse_args(['--from', '1', '--to', '1', '--source', 'scale-replay']))
        self.assertEqual(Path(replay[0][1]).name, 'extract_scale_order_params.py')
        self.assertEqual(replay[0][replay[0].index('--limit') + 1], '0')

    def test_hashing_stable_across_processes(self):
        code = "from mas_collapse.embed.backend import HashingEmbedder; print(HashingEmbedder(8).embed(['same words']).tolist())"
        one = subprocess.check_output([sys.executable, '-c', code], cwd=ROOT)
        two = subprocess.check_output([sys.executable, '-c', code], cwd=ROOT)
        self.assertEqual(one, two)
        self.assertEqual(HashingEmbedder(8).embed(['same words']).shape, (1, 8))


class ReplayTests(unittest.TestCase):
    def test_alignment_and_failure_detection(self):
        human = [{'position': i, 'author': 'fixture', 'alias': 'A', 'comment_id': str(i), 'body': 'human text'} for i in range(2)]
        sim = [{**r, 'body': 'sim text', 'shown_indices': list(range(i))} for i, r in enumerate(human)]
        thread = {'post_id': 'fixture', 'turns': human}
        meta = {'type': 'meta', 'post_id': 'fixture', 'n_turns': 2}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.jsonl'
            def write(rows): path.write_text('\n'.join(json.dumps(r) for r in [meta] + rows), encoding='utf-8')
            write(sim); self.assertEqual(len(validate_pair(thread, path)[0]), 2)
            mutations = [('body', ''), ('author', 'wrong'), ('position', 99), ('shown_indices', [1]), ('error', 'failed')]
            for key, value in mutations:
                bad = copy.deepcopy(sim); bad[1][key] = value; write(bad)
                with self.subTest(key=key), self.assertRaises(ValueError): validate_pair(thread, path)
            write(sim[:1])
            with self.assertRaises(ValueError): validate_pair(thread, path)


if __name__ == '__main__':
    unittest.main()
