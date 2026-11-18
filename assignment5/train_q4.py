import argparse
import datetime
import gc
import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_train_val_loaders
from models_q4 import ARCH, build_model, model_config
from train import sample_points, test as validate, train as train_one_epoch
from utils import checkpoint_config, create_dir, write_json

DEFAULT_BATCH_SIZE = {"cls": 16, "seg": 8}
RESUME_KEYS = ("task", "num_points", "batch_size", "lr", "k", "seed", "val_frac", "num_cls_class", "num_seg_class", "main_dir")


def run_name(task, num_points, tag=""):
    return "{}_n{}".format(task, num_points) + ("_" + tag if tag else "")


def create_parser(prog=None, mode="train"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Q4 Point Transformer {}. Writes only under --checkpoint_root, --log_root and output/q4.".format(
            "training" if mode == "train" else "training-step benchmark on disposable state (no checkpoints, no TensorBoard)"))
    parser.add_argument('--task', choices=["cls", "seg", "all"] if mode == "train" else ["cls", "seg"], required=True)
    parser.add_argument('--num_points', type=int, default=10000, help='Points per object (e.g. 10000 or 2500)')
    parser.add_argument('--batch_size', type=int, default=None, help='Default: cls 16, seg 8')
    parser.add_argument('--k', type=int, default=16, help='kNN neighborhood size (clamped to the available points)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_cls_class', type=int, default=3)
    parser.add_argument('--num_seg_class', type=int, default=6)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--tag', type=str, default='', help='Suffix that keeps a non-default configuration in its own directories')
    if mode == "train":
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/q4')
        parser.add_argument('--log_root', type=str, default='./logs/q4')
        parser.add_argument('--checkpoint_every', type=int, default=10)
        parser.add_argument('--overwrite', action='store_true', help='Start fresh even if this configuration already has checkpoints')
        parser.add_argument('--resume', action='store_true', help='Continue from <run>/last_model.pt (model, optimizer, epoch, best, RNG)')
    else:
        parser.add_argument('--warmup_batches', type=int, default=1, choices=[0, 1])
        parser.add_argument('--timed_batches', type=int, default=2, choices=[1, 2])
        parser.add_argument('--output_dir', type=str, default='./output/q4/benchmark')
    return parser


def task_args(args, task):
    a = argparse.Namespace(**vars(args))
    a.task = task
    a.batch_size = args.batch_size or DEFAULT_BATCH_SIZE[task]
    a.device = torch.device(args.device)
    a.arch = ARCH
    a.run_name = run_name(task, args.num_points, args.tag)
    return a


def num_classes(args):
    return args.num_cls_class if args.task == "cls" else args.num_seg_class


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def checkpoint_dict(model, args, epoch, val_acc):
    return dict(arch=ARCH, model_config=model.config, task=args.task, num_classes=num_classes(args), epoch=epoch,
                val_acc=val_acc, config=checkpoint_config(args), model_state_dict=model.state_dict())


def train_task(args):
    run_dir = os.path.join(args.checkpoint_root, args.run_name)
    args.checkpoint_dir = run_dir
    last_path, best_path = os.path.join(run_dir, "last_model.pt"), os.path.join(run_dir, "best_model.pt")
    state = None
    if args.resume:
        if not os.path.isfile(last_path):
            sys.exit("--resume: {} not found".format(last_path))
        state = torch.load(last_path, map_location="cpu")
        if state.get("arch") != ARCH:
            sys.exit("--resume: {} is not a {} checkpoint".format(last_path, ARCH))
        mismatched = {k: (state["config"].get(k), getattr(args, k)) for k in RESUME_KEYS if state["config"].get(k) != getattr(args, k)}
        if mismatched:
            sys.exit("--resume: settings differ from the checkpoint (checkpoint, requested): {}".format(mismatched))
    elif (os.path.exists(best_path) or os.path.exists(last_path)) and not args.overwrite:
        sys.exit("{} already has checkpoints; pass --resume to continue, --overwrite to start fresh, or --tag for a separate run".format(run_dir))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    create_dir(run_dir)
    model = build_model(model_config(args.task, num_classes(args), args.k)).to(args.device)
    opt = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))
    train_loader, val_loader, train_idx, val_idx, num_available = get_train_val_loaders(args)
    if not 0 < args.num_points <= num_available:
        sys.exit("--num_points must be in [1, {}], got {}".format(num_available, args.num_points))

    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        start_epoch, best_acc, best_epoch = state["epoch"] + 1, state["best_val_acc"], state["best_epoch"]
        history, log_dir = state["history"], state["log_dir"]
        torch.set_rng_state(state["torch_rng_state"])
        if state.get("cuda_rng_state") is not None and args.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng_state"])
        print("resuming {} from epoch {} (best val acc {:.4f} at epoch {})".format(last_path, start_epoch, best_acc, best_epoch))
    else:
        if args.overwrite and os.path.isdir(run_dir) and os.listdir(run_dir):
            print("warning: --overwrite reuses {}; files from the previous run are replaced as new ones are written".format(run_dir))
        start_epoch, best_acc, best_epoch = 0, -1.0, None
        log_dir = os.path.join(args.log_root, args.run_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        history = {
            "arch": ARCH, "task": args.task, "num_points": args.num_points, "batch_size": args.batch_size, "lr": args.lr,
            "selection": "best_model.pt = highest validation accuracy (ties -> later epoch); last_model.pt = latest epoch",
            "val_frac": args.val_frac, "seed": args.seed, "num_train": len(train_idx), "num_val": len(val_idx),
            "val_indices": val_idx.tolist(), "log_dir": log_dir, "epochs": [],
        }
    writer = SummaryWriter(log_dir)

    print("Q4 {} | {} params | {} points | batch {} | {} train / {} val objects | {} train batches per epoch".format(
        args.run_name, sum(p.numel() for p in model.parameters()), args.num_points, args.batch_size,
        len(train_idx), len(val_idx), len(train_loader)))
    print("checkpoints: {}  tensorboard: {}".format(run_dir, log_dir))
    for epoch in range(start_epoch, args.num_epochs):
        start = time.time()
        train_loss = train_one_epoch(train_loader, model, opt, epoch, args, writer)
        val_acc = validate(val_loader, model, epoch, args, writer)
        seconds = time.time() - start
        improved = val_acc >= best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch
            save_atomic(checkpoint_dict(model, args, epoch, val_acc), best_path)
        if epoch % args.checkpoint_every == 0:
            save_atomic(checkpoint_dict(model, args, epoch, val_acc), os.path.join(run_dir, "model_epoch_{}.pt".format(epoch)))
        history["epochs"].append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc, "seconds": seconds})
        history.update(best_epoch=best_epoch, best_val_acc=best_acc)
        last = checkpoint_dict(model, args, epoch, val_acc)
        last.update(optimizer_state_dict=opt.state_dict(), best_val_acc=best_acc, best_epoch=best_epoch, history=history,
                    log_dir=log_dir, torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state() if args.device.type == "cuda" else None)
        save_atomic(last, last_path)
        write_json(os.path.join(run_dir, "history.json"), history)
        print("epoch {:3d}  train loss {:.4f}  val acc {:.4f}  {:.0f}s{}".format(
            epoch, train_loss, val_acc, seconds, "  (best)" if improved else ""))
    writer.close()
    print("Q4 {} done: best val acc {:.4f} at epoch {}".format(args.run_name, best_acc, best_epoch))


def benchmark_once(args, batch_size):
    args = argparse.Namespace(**vars(args))
    args.batch_size = batch_size
    torch.manual_seed(args.seed)
    train_loader = get_train_val_loaders(args)[0]
    model = build_model(model_config(args.task, num_classes(args), args.k)).to(args.device).train()
    opt = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))
    criterion = torch.nn.CrossEntropyLoss()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    batches = iter(train_loader)
    step_ms, load_ms = [], []
    for step in range(args.warmup_batches + args.timed_batches):
        t0 = time.perf_counter()
        point_clouds, labels = next(batches)
        point_clouds, labels = sample_points(point_clouds.to(args.device).float(), labels.to(args.device).long(), args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        predictions = model(point_clouds)
        loss = criterion(predictions.reshape(-1, predictions.shape[-1]), labels.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        end.record()
        torch.cuda.synchronize()
        if step >= args.warmup_batches:
            step_ms.append(start.elapsed_time(end))
            load_ms.append((t1 - t0) * 1000)
    mean_ms = float(np.mean(step_ms))
    return {
        "disposable_state": True,
        "task": args.task, "arch": ARCH, "num_points": args.num_points, "batch_size": batch_size, "k": args.k,
        "warmup_batches": args.warmup_batches, "timed_batches": args.timed_batches,
        "timed_step_ms": step_ms, "mean_step_ms": mean_ms,
        "timing_scope": "CUDA events around forward + loss + zero_grad + backward + optimizer.step; excludes data loading",
        "data_loading_and_h2d_ms": load_ms,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
        "train_batches_per_epoch": len(train_loader),
        "rough_epoch_train_minutes": mean_ms * len(train_loader) / 60000,
        "rough_total_train_hours": mean_ms * len(train_loader) * args.num_epochs / 3.6e6,
        "num_epochs": args.num_epochs,
        "estimate_note": "rough extrapolation from {} timed batch(es), not a measured epoch; validation, checkpointing, "
                         "data loading, logging, startup and timing variability add overhead".format(args.timed_batches),
        "device": torch.cuda.get_device_name(),
        "lr": args.lr,
        "seed": args.seed,
    }


def benchmark_task(args):
    if args.device.type != "cuda":
        sys.exit("benchmark requires a CUDA device")
    result, attempts = None, []
    for batch_size in dict.fromkeys([args.batch_size, max(1, args.batch_size // 2)]):
        try:
            result = benchmark_once(args, batch_size)
            break
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch_size": batch_size, "result": "out of memory"})
            print("out of memory at batch size {}".format(batch_size))
        gc.collect()
        torch.cuda.empty_cache()
    if result is None:
        sys.exit("benchmark failed: {}".format(attempts))
    result["failed_attempts"] = attempts
    create_dir(args.output_dir)
    path = os.path.join(args.output_dir, "{}_bs{}.json".format(args.run_name, result["batch_size"]))
    write_json(path, result)
    print("Q4 benchmark {} (disposable weights, not a trained result)".format(args.run_name))
    print("  batch size {}  timed steps (ms) {}  mean {:.1f} ms  [{}]".format(
        result["batch_size"], ", ".join("{:.1f}".format(t) for t in result["timed_step_ms"]), result["mean_step_ms"], result["timing_scope"]))
    print("  data loading + host-to-device (ms, not included) {}".format(", ".join("{:.1f}".format(t) for t in result["data_loading_and_h2d_ms"])))
    print("  peak allocated {:.2f} GiB  peak reserved {:.2f} GiB".format(result["peak_allocated_gib"], result["peak_reserved_gib"]))
    print("  rough estimate: {} batches/epoch -> ~{:.1f} min/epoch training, ~{:.1f} h for {} epochs ({})".format(
        result["train_batches_per_epoch"], result["rough_epoch_train_minutes"], result["rough_total_train_hours"],
        args.num_epochs, result["estimate_note"]))
    print("  -> {}".format(path))
    return result


def main(argv=None, prog=None, mode="train"):
    args = create_parser(prog, mode).parse_args(argv)
    for task in (["cls", "seg"] if args.task == "all" else [args.task]):
        (train_task if mode == "train" else benchmark_task)(task_args(args, task))


if __name__ == '__main__':
    argv = sys.argv[1:]
    mode = argv.pop(0) if argv[:1] in (["train"], ["benchmark"]) else "train"
    main(argv, prog="train_q4.py " + mode, mode=mode)
