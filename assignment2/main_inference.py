"""Unified evaluation / visualization entry point for Assignment 2.

A thin dispatcher: every target runs an existing script (q1_viz.py, q2_viz.py, q2_5.py,
q3_1.py, q3_2.py, q3_3.py) as a subprocess with the same Python interpreter, from this
directory. Inference only reads checkpoints; outputs go to the scripts' usual folders.

    python main_inference.py                  # same as --target all
    python main_inference.py --target vox
    python main_inference.py --target point500
    python main_inference.py --target all --dry_run
"""
import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ALL_ORDER = ['q1_vox', 'q1_point', 'q1_mesh', 'vox', 'point', 'mesh', 'point500',
             'q2_5', 'q3_1', 'q3_2', 'q3_3', 'q3_3_compare']

# target -> (script, args before --device, args after --device, output dir)
TARGETS = {
    'q1_vox': ('q1_viz.py', ['--type', 'vox'], [], 'outputs/q1'),
    'q1_point': ('q1_viz.py', ['--type', 'point'], [], 'outputs/q1'),
    'q1_mesh': ('q1_viz.py', ['--type', 'mesh'], [], 'outputs/q1'),
    'vox': ('q2_viz.py', ['--type', 'vox'], ['--load_checkpoint'], 'outputs/q2'),
    'point': ('q2_viz.py', ['--type', 'point'], ['--load_checkpoint'], 'outputs/q2'),
    'mesh': ('q2_viz.py', ['--type', 'mesh'], ['--load_checkpoint'], 'outputs/q2'),
    'point500': ('q2_viz.py', ['--type', 'point'],
                 ['--checkpoint', 'checkpoint_point_n500.pth', '--n_points', '500', '--load_checkpoint',
                  '--output_dir', 'outputs/q2_4/n500'],
                 'outputs/q2_4/n500'),
    'q2_5': ('q2_5.py', [], [], 'outputs/q2_5'),
    'q3_1': ('q3_1.py', ['--mode', 'eval'], ['--load_checkpoint'], 'outputs/q3_1'),
    'q3_2': ('q3_2.py', ['--mode', 'eval'], ['--load_checkpoint'], 'outputs/q3_2'),
    'q3_3': ('q3_3.py', ['--mode', 'eval'], ['--load_checkpoint'], 'outputs/q3_3'),
    'q3_3_compare': ('q3_3.py', ['--mode', 'compare'], [], 'outputs/q3_3'),
}


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Launch Assignment 2 evaluation / visualization jobs '
                    '(dispatches to the existing scripts).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='targets:\n'
               '  q1_vox        Q1 voxel fitting        q1_viz.py --type vox      -> outputs/q1\n'
               '  q1_point      Q1 point fitting        q1_viz.py --type point    -> outputs/q1\n'
               '  q1_mesh       Q1 mesh fitting         q1_viz.py --type mesh     -> outputs/q1\n'
               '  vox           Part 2 voxel            q2_viz.py --type vox      -> outputs/q2\n'
               '  point         Part 2 point cloud      q2_viz.py --type point    -> outputs/q2\n'
               '  mesh          Part 2 mesh             q2_viz.py --type mesh     -> outputs/q2\n'
               '  point500      Q2.4 n_points=500       q2_viz.py (n500 ckpt)     -> outputs/q2_4/n500\n'
               '  q2_5          Q2.5 interpretability   q2_5.py                   -> outputs/q2_5\n'
               '  q3_1          Q3.1 implicit           q3_1.py --mode eval       -> outputs/q3_1\n'
               '  q3_2          Q3.2 parametric         q3_2.py --mode eval       -> outputs/q3_2\n'
               '  q3_3          Q3.3 three-class        q3_3.py --mode eval       -> outputs/q3_3\n'
               '  q3_3_compare  Q3.3 chair-only vs 3c   q3_3.py --mode compare    -> outputs/q3_3\n'
               '  all           the above, sequentially, in this order (default)\n\n'
               'Note: q1_* re-run the Q1 optimization (fit_data.py defaults) before rendering.')
    parser.add_argument('--target', default='all', choices=list(TARGETS) + ['all'],
                        help='which evaluation / visualization to run (default: all)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='passed to the child script as --device (default: cuda)')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the commands that would run without executing them')
    return parser


def build_command(target, args):
    script, pre, post, _ = TARGETS[target]
    return [sys.executable, script] + pre + ['--device', args.device] + post


def display(cmd):
    return shlex.join(['python' if cmd[0] == sys.executable else cmd[0]] + cmd[1:])


def main():
    args = get_args_parser().parse_args()
    targets = ALL_ORDER if args.target == 'all' else [args.target]

    for target in targets:
        script = os.path.join(HERE, TARGETS[target][0])
        if not os.path.isfile(script):
            print(f'ERROR: {script} not found', file=sys.stderr)
            return 1

    for target in targets:
        cmd = build_command(target, args)
        print('=' * 40)
        print(f'Running inference target: {target}')
        print('=' * 40)
        print('Command:')
        print(display(cmd))
        print(f'Output directory: {TARGETS[target][3]}', flush=True)
        if args.dry_run:
            print('[dry run] not executed\n')
            continue
        try:
            subprocess.run(cmd, cwd=HERE, check=True)
        except subprocess.CalledProcessError as e:
            print(f'\nInference target {target} failed with exit code {e.returncode}', file=sys.stderr)
            return e.returncode or 1
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
