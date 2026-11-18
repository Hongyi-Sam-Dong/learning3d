import os
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import cls_model, seg_model
from data_loader import get_train_val_loaders
from utils import save_checkpoint, create_dir, load_weights, write_json, save_checkpoint_dir_guard

def sample_points(point_clouds, labels, args, generator=None):
    if args.num_points >= point_clouds.shape[1]:
        return point_clouds, labels
    idx = torch.randperm(point_clouds.shape[1], generator=generator)[:args.num_points].to(point_clouds.device)
    if args.task == "seg":
        labels = labels[:, idx]
    return point_clouds[:, idx], labels

def train(train_dataloader, model, opt, epoch, args, writer):

    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        point_clouds, labels = batch
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)
        point_clouds, labels = sample_points(point_clouds, labels, args)

        # ------ TO DO: Forward Pass ------
        predictions = model(point_clouds)

        if (args.task == "seg"):
            labels = labels.reshape([-1])
            predictions = predictions.reshape([-1, args.num_seg_class])

        # Compute Loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)
        epoch_loss += loss.item()

        # Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar('train_loss', loss.item(), step+i)

    return epoch_loss / len(train_dataloader)

def test(test_dataloader, model, epoch, args, writer):

    model.eval()
    generator = torch.Generator().manual_seed(args.seed)

    # Evaluation in Classification Task
    if (args.task == "cls"):
        correct_obj = 0
        num_obj = 0
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)
            point_clouds, labels = sample_points(point_clouds, labels, args, generator)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                pred_labels = model(point_clouds).argmax(dim=-1)
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_obj / num_obj


    # Evaluation in Segmentation Task
    else:
        correct_point = 0
        num_point = 0
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)
            point_clouds, labels = sample_points(point_clouds, labels, args, generator)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                pred_labels = model(point_clouds).argmax(dim=-1)

            correct_point += pred_labels.eq(labels.data).cpu().sum().item()
            num_point += labels.view([-1,1]).size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_point / num_point

    writer.add_scalar("val_acc", accuracy, epoch)
    return accuracy


def main(args):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create Directories
    create_dir(args.checkpoint_dir)
    if not args.load_checkpoint:
        save_checkpoint_dir_guard(os.path.join(args.checkpoint_dir, 'best_model.pt'), args.overwrite)
    create_dir(args.log_dir)

    # Tensorboard Logger
    writer = SummaryWriter(os.path.join(args.log_dir, args.task+"_"+args.exp_name))

    # ------ TO DO: Initialize Model ------
    if args.task == "cls":
        model = cls_model(num_classes=args.num_cls_class).to(args.device)
    else:
        model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)

    # Load Checkpoint 
    if args.load_checkpoint:
        model_path = "{}/{}.pt".format(args.checkpoint_dir,args.load_checkpoint)
        load_weights(model, model_path, args.device)
        print ("successfully loaded checkpoint from {}".format(model_path))

    # Optimizer
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    # Dataloader for Training & Testing
    train_dataloader, val_dataloader, train_idx, val_idx, num_available = get_train_val_loaders(args)
    if not 0 < args.num_points <= num_available:
        raise SystemExit("--num_points must be in [1, {}], got {}".format(num_available, args.num_points))

    print ("successfully loaded data: {} train / {} val objects".format(len(train_idx), len(val_idx)))

    best_acc = -1
    best_epoch = None
    history = {
        "task": args.task,
        "selection": "best_model.pt = highest validation accuracy (ties -> later epoch); last_model.pt = latest epoch",
        "val_frac": args.val_frac,
        "seed": args.seed,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "val_indices": val_idx.tolist(),
        "epochs": [],
    }

    print ("======== start training for {} task ========".format(args.task))
    print ("(check tensorboard for plots of experiment {})".format(os.path.join(args.log_dir, args.task+"_"+args.exp_name)))

    for epoch in range(args.num_epochs):

        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)

        # Test
        current_acc = test(val_dataloader, model, epoch, args, writer)

        print ("epoch: {}   train loss: {:.4f}   val accuracy: {:.4f}".format(epoch, train_epoch_loss, current_acc))

        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            print ("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=False, extra={"val_acc": current_acc})

        # Save Best Model Checkpoint
        if (current_acc >= best_acc):
            best_acc = current_acc
            best_epoch = epoch
            print ("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=True, extra={"val_acc": current_acc})

        save_checkpoint(epoch=epoch, model=model, args=args, name="last_model", extra={"val_acc": current_acc})
        history["epochs"].append({"epoch": epoch, "train_loss": train_epoch_loss, "val_acc": current_acc})
        history.update(best_epoch=best_epoch, best_val_acc=best_acc)
        write_json(os.path.join(args.checkpoint_dir, "history.json"), history)

    writer.close()
    print ("======== training completes: best val accuracy {:.4f} at epoch {} ========".format(best_acc, best_epoch))


def create_parser(prog=None):
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(prog=prog)

    # Model & Data hyper-parameters
    parser.add_argument('--task', type=str, default="cls", choices=["cls", "seg", "all"], help='The task: cls, seg, or all (cls then seg)')
    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classification classes')
    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='Points per object used in training/validation (random subset per batch if < available)')

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (default 0.001)')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Fraction of the training set held out for checkpoint selection (stratified by class for cls)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoints go to <checkpoint_dir>/<task>')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--checkpoint_every', type=int , default=10)

    parser.add_argument('--load_checkpoint', type=str, default='')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing an existing <checkpoint_dir>/<task>/best_model.pt')


    return parser


def run(args):
    for task in (["cls", "seg"] if args.task == "all" else [args.task]):
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task
        task_args.device = torch.device(args.device)
        task_args.checkpoint_dir = os.path.join(args.checkpoint_dir, task) # checkpoint directory is task specific
        main(task_args)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run(args)
