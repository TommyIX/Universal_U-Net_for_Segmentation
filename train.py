import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ADMIRE_Dataset as Dataset
from logger import Logger
from loss import DiceLoss
from PIL import Image, ImageDraw
from unet import UNet
from utils import log_images, dsc


def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=3, out_channels=1)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    logger = Logger(args.logs)
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in range(args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []
            print("\nepoch: %d, %s: "%(epoch, phase))

            for i, data in tqdm(enumerate(loaders[phase]),total=len(loaders[phase])):

                if phase == "train":
                    step += 1

                x, y_true = data
                x2 = torch.transpose(x,1,3)
                x3 = torch.transpose(x2,2,3)
                x = x3
                y_true = y_true.unsqueeze(1)
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                            if i * args.batch_size < args.vis_images:
                                tag = "image/{}".format(i)
                                num_images = args.vis_images - i * args.batch_size
                                logger.image_list_summary(
                                    tag,
                                    log_images(x, y_true, y_pred)[:num_images],
                                    step,
                                )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                gen_thepic(validation_pred,validation_true,epoch)
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                    )
                )

                logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet_try300.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid

def datasets(args):
    # train = Dataset(
    #     images_dir=args.images,
    #     subset="train",
    #     image_size=args.image_size,
    #     transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    # )
    # valid = Dataset(
    #     images_dir=args.images,
    #     subset="validation",
    #     image_size=args.image_size,
    #     random_sampling=False,
    # )

    train = Dataset(
        imgsize=args.image_size,
        folder_path=args.images,
        subset = "train",
        fold_num=0,
    )
    valid = Dataset(
        imgsize=args.image_size,
        folder_path=args.images,
        subset = "validation",
        fold_num=0,
    )

    return train, valid

def gen_thepic(validation_pred, validation_true, epo):
    for p in range(len(validation_pred)):
        # y_pred = np.round(np.array(validation_pred[p]).squeeze()*255).astype(int)
        y_pred = np.round(np.array(validation_pred[p]).squeeze())*255
        # y_true = np.round(np.array(validation_true[p]).squeeze()*255).astype(int)

        y_predimg = Image.fromarray(y_pred)
        y_predimg.save("results/pred_epo"+str(epo)+"_val"+str(p)+".gif")

        # predimgnp = np.array(Image.fromarray(y_pred).convert('RGB'))
        # predimgnp[:,:,1] = 0
        # predimgnp[:, :, 2] = 0
        # predtrunp = np.array(Image.fromarray(y_true).convert('RGB'))
        # predtrunp[:,:,0] = 0
        # predtrunp[:,:,2] = 0
        #
        # predfin = (predimgnp*0.5+predtrunp).astype(np.uint8)
        # predfin = Image.fromarray(np.clip(predfin,0,255),"RGB")
        # predfin.save("results/epo"+str(epo)+"_val"+str(p)+".jpg")


def dsc_per_volume(validation_pred, validation_true):
    dsc_list = []
    for p in range(len(validation_pred)):
        y_pred = np.array(validation_pred[p]).squeeze()
        y_true = np.array(validation_true[p]).squeeze()
        dsc_list.append(dsc(y_pred, y_true))
    return dsc_list


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of workers for data loading (default: 0)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="./data", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    args = parser.parse_args()
    main(args)
