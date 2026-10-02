"""Unified training entry point for Assignment 2.

A thin dispatcher: every target runs an existing script (train_model.py, q3_1.py, q3_2.py,
q3_3.py) as a subprocess with the same Python interpreter, from this directory.

    python main_train.py                      # same as --target all
    python main_train.py --target q3_1
    python main_train.py --target vox --fresh --overwrite
    python main_train.py --dry_run

Checkpoint handling (automatic):
    * If a target's checkpoint exists, training resumes from it (--load_checkpoint is passed).
      If it does not exist, training starts from scratch.
    * --fresh forces a from-scratch run. If that would overwrite an existing checkpoint it is
      refused unless --overwrite is also given. For --target all every target is checked before
      anything starts.
    * point500: train_model.py always writes checkpoint_{type}.pth in its working directory,
      i.e. checkpoint_point.pth. To keep checkpoint_point.pth untouched, train_model.py is run
      inside a fresh temporary directory under this folder; only after it exits successfully is
      the resulting checkpoint moved to checkpoint_point_n500.pth. When resuming,
      checkpoint_point_n500.pth is first copied into that directory as checkpoint_point.pth.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

ALL_ORDER = ['vox', 'point', 'mesh', 'point500', 'q3_1', 'q3_2', 'q3_3']

# target -> (display name, script, fixed args, checkpoint the job writes)
TARGETS = {
    'vox': (
        'voxel',
        'train_model.py',
        ['--type', 'vox', '--max_iter', '10001'],
        'checkpoint_vox.pth'
    ),

    'point': (
        'point',
        'train_model.py',
        ['--type', 'point', '--max_iter', '10001'],
        'checkpoint_point.pth'
    ),

    'mesh': (
        'mesh',
        'train_model.py',
        ['--type', 'mesh', '--max_iter', '10001'],
        'checkpoint_mesh.pth'
    ),

    'point500': (
        'point500',
        'train_model.py',
        ['--type', 'point',
         '--n_points', '500',
         '--max_iter', '10001',
         '--save_freq', '2000'],
        'checkpoint_point_n500.pth'
    ),

    'q3_1': (
        'q3_1',
        'q3_1.py',
        ['--mode', 'train', '--max_iter', '10001'],
        'checkpoint_q3_1.pth'
    ),

    'q3_2': (
        'q3_2',
        'q3_2.py',
        ['--mode', 'train', '--max_iter', '10001'],
        'checkpoint_q3_2.pth'
    ),

    'q3_3': (
        'q3_3',
        'q3_3.py',
        ['--mode', 'train', '--max_iter', '10001'],
        'checkpoint_q3_3_vox.pth'
    ),
}

# train_model.py writes this name (relative to its cwd) for --type point.
POINT500_RAW_CHECKPOINT = 'checkpoint_point.pth'


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Launch Assignment 2 training jobs (dispatches to the existing scripts).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='targets:\n'
               '  vox       Part 2 voxel        train_model.py --type vox      -> checkpoint_vox.pth\n'
               '  point     Part 2 point cloud  train_model.py --type point    -> checkpoint_point.pth\n'
               '  mesh      Part 2 mesh         train_model.py --type mesh     -> checkpoint_mesh.pth\n'
               '  point500  Q2.4 n_points=500   train_model.py (isolated cwd)  -> checkpoint_point_n500.pth\n'
               '  q3_1      Q3.1 implicit       q3_1.py --mode train           -> checkpoint_q3_1.pth\n'
               '  q3_2      Q3.2 parametric     q3_2.py --mode train           -> checkpoint_q3_2.pth\n'
               '  q3_3      Q3.3 three-class    q3_3.py --mode train           -> checkpoint_q3_3_vox.pth\n'
               '  all       the above, sequentially, in this order (default)\n\n'
               'An existing checkpoint is resumed automatically; a missing one starts from scratch.')
    parser.add_argument('--target', default='all', choices=list(TARGETS) + ['all'],
                        help='which model to train (default: all)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='passed to the child script as --device (default: cuda)')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='DataLoader workers, passed to the child script as --num_workers '
                             '(default: 16; the R2N2 loader is CPU-bound on OBJ parsing)')
    parser.add_argument('--fresh', action='store_true',
                        help='train from scratch instead of resuming an existing checkpoint '
                             '(requires --overwrite if that checkpoint exists)')
    parser.add_argument('--overwrite', action='store_true',
                        help='confirm that --fresh may replace an existing checkpoint')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the commands that would run without executing them')
    return parser


def checkpoint_path(target):
    return os.path.join(HERE, TARGETS[target][3])


def should_resume(target, args):
    """Resume whenever the target's checkpoint exists, unless --fresh was given."""
    return os.path.isfile(checkpoint_path(target)) and not args.fresh


def build_command(target, args, resume):
    _, script, fixed, _ = TARGETS[target]
    cmd = [sys.executable, os.path.join(HERE, script) if target == 'point500' else script]
    cmd += fixed + ['--device', args.device, '--num_workers', str(args.num_workers)]
    if resume:
        cmd.append('--load_checkpoint')
    return cmd


def display(cmd):
    shown = ['python' if cmd[0] == sys.executable else cmd[0]]
    shown += [os.path.relpath(c, HERE) if os.path.isabs(c) and c.startswith(HERE) else c for c in cmd[1:]]
    return shlex.join(shown)


def check_target(target, args):
    """Return a list of reasons the target must not run (empty = safe)."""
    problems = []
    if args.fresh and os.path.isfile(checkpoint_path(target)) and not args.overwrite:
        problems.append(f'--fresh would overwrite {TARGETS[target][3]} '
                        '(add --overwrite to confirm, or drop --fresh to resume it)')
    return problems


def banner(name):
    print('=' * 40)
    print(f'Running training target: {name}')
    print('=' * 40, flush=True)


def run_point500(cmd, resume):
    final = os.path.join(HERE, TARGETS['point500'][3])
    protected = os.path.join(HERE, POINT500_RAW_CHECKPOINT)
    before = os.stat(protected) if os.path.isfile(protected) else None

    workdir = tempfile.mkdtemp(prefix='.point500_run_', dir=HERE)
    print(f'Isolated working directory: {os.path.relpath(workdir, HERE)}')
    if resume:
        shutil.copy2(final, os.path.join(workdir, POINT500_RAW_CHECKPOINT))
        print(f'Copied {TARGETS["point500"][3]} -> {os.path.relpath(workdir, HERE)}/{POINT500_RAW_CHECKPOINT}')

    subprocess.run(cmd, cwd=workdir, check=True)

    produced = os.path.join(workdir, POINT500_RAW_CHECKPOINT)
    if not os.path.isfile(produced):
        raise RuntimeError(f'train_model.py finished but wrote no checkpoint in {workdir}')
    os.replace(produced, final)
    print(f'Moved {os.path.relpath(produced, HERE)} -> {TARGETS["point500"][3]}')
    shutil.rmtree(workdir)

    after = os.stat(protected) if os.path.isfile(protected) else None
    if (before is None) != (after is None) or (
            before and (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)):
        raise RuntimeError(f'{POINT500_RAW_CHECKPOINT} changed during the point500 run')


def main():
    args = get_args_parser().parse_args()
    targets = ALL_ORDER if args.target == 'all' else [args.target]

    for target in targets:
        script = os.path.join(HERE, TARGETS[target][1])
        if not os.path.isfile(script):
            print(f'ERROR: {script} not found', file=sys.stderr)
            return 1

    # Safety checks for every target up front, so `all` never stops hours in.
    blocked = {t: p for t in targets if (p := check_target(t, args))}
    for t, problems in blocked.items():
        for p in problems:
            print(f'{"WOULD REFUSE" if args.dry_run else "REFUSING"} {t}: {p}', file=sys.stderr)
    if blocked and not args.dry_run:
        return 1

    for target in targets:
        name = TARGETS[target][0]
        resume = should_resume(target, args)
        cmd = build_command(target, args, resume)
        banner(name)
        ckpt = os.path.relpath(checkpoint_path(target), HERE)
        if resume:
            print('AUTO-RESUME: checkpoint found')
            print(f'Found checkpoint: {ckpt}')
            print('Resuming training automatically.')
        else:
            print(f'START-FRESH: {"--fresh given" if args.fresh else "checkpoint not found"}')
            print('No checkpoint found.' if not os.path.isfile(checkpoint_path(target))
                  else f'Ignoring existing checkpoint {ckpt} (--fresh --overwrite).')
            print('Starting training from scratch.')
        if target == 'point500':
            print('(run in an isolated temporary directory; result moved to '
                  f'{TARGETS[target][3]} on success)')
        print('Command:')
        print(display(cmd), flush=True)
        if args.dry_run:
            print('[dry run] not executed\n')
            continue
        try:
            if target == 'point500':
                run_point500(cmd, resume)
            else:
                subprocess.run(cmd, cwd=HERE, check=True)
        except subprocess.CalledProcessError as e:
            print(f'\nTraining target {target} failed with exit code {e.returncode}', file=sys.stderr)
            return e.returncode or 1
        except (RuntimeError, OSError) as e:
            print(f'\nTraining target {target} failed: {e}', file=sys.stderr)
            return 1
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
