from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


import time

import misc.utils as utils
import wandb
import time
import os
import random
import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf 
import json

from misc.get_splits import get_dataset_splits
from data.datasets import LSMDCDataset
from data_loaders import LSMDCDataloader
from trainer import train_model_fitb, eval_model_fitb, evaluate_caption, train_eval_model_caption, train_model_fc, train_model_joint

from models.mst_model import MSTModel

from misc.run_checkpoints import fitb_ckpt, fc_ckpt


torch.backends.cudnn.enabled = False
torch.autograd.set_detect_anomaly(True)


def seed_everything(track, seed=0):
    print(f"Random seed value is : {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(opt):

    # Tracking_mode
    enable_tracking = opt.track
    is_overfit = opt.overfit
    use_checkpoint = opt.use_checkpoint

    task_list = ["fitb", "fc", "joint"]

    fitb = opt.run_type == task_list[0] or opt.run_type == task_list[2]
    fc = opt.run_type == task_list[1] or opt.run_type == task_list[2]
    joint = opt.run_type == task_list[2]

    split_info, split_ix, split_map = get_dataset_splits(opt)



    # For fillin task
    for x in split_info.keys():
        split_info[x] = [i for i in split_info[x] if len(i["blank_indexes"]) != 0]

    print("DataLoader loading json file: ", opt.input_json)
    info = json.load(open(os.path.join(opt.data_dir, opt.input_json)))

    # # Splits are : "train", "val", "overfit"
    if is_overfit:
        train_dataset = LSMDCDataset(opt, split_info["overfit"], split_ix, "overfit", split_map, info)
    else:
        train_dataset = LSMDCDataset(opt, split_info["train"], split_ix, "train", split_map, info)
    
    val_dataset = LSMDCDataset(opt, split_info["val"], split_ix, "val", split_map, info)
    
    start_time = time.time()
    loader = LSMDCDataloader(train_dataset,batch_size=opt.batch_size)
    train_dataloader = loader.get_loader()
    end_time = time.time()

    val_loader = LSMDCDataloader(val_dataset,batch_size=opt.batch_size, shuffle=False)
    val_dataloader = val_loader.get_loader()


    eval_kwargs = {
            "split": "val",
            "eval_accuracy": opt.eval_accuracy,
            "id": opt.val_id,
            "remove": 1,
        }

    opt.unique_characters = train_dataset.unique_characters

    print(f"Time taken to init DataLoader : {end_time - start_time}")

    start_time = time.time()
    mst_model = MSTModel(opt)
    mst_model = mst_model.cuda()
    end_time = time.time()

    print(f"Time taken to init MSTModel : {end_time - start_time}")


    mst_model.train()
  
    mst_optimizer = utils.build_optimizer(mst_model.parameters(), opt)
    epochs = opt.pre_nepoch + 1

    vocab = mst_model.tokenizer


    if use_checkpoint:

        print("Using checkpoint")
        mst_model.load_state_dict(torch.load(os.path.join(opt.data_dir, opt.ckpt_path)))
        mst_model.eval()

        if fitb:
            fitb_ckpt(mst_model, val_dataloader, eval_kwargs, enable_tracking)
        
        if fc:
            fc_ckpt(mst_model, val_dataloader, vocab, enable_tracking)
        print("Done!")
        return


    if not is_overfit:

        accuracy = None
        cap_metrics = None
        if fitb:
            val_fitb_loss, _, accuracy, _ = eval_model_fitb(
                            mst_model, val_dataloader, eval_kwargs=eval_kwargs
                        )
        
        if fc:
            val_fc_loss = train_eval_model_caption(mst_model, val_dataloader) 
            cap_metrics = evaluate_caption(mst_model, val_dataloader, vocab, epoch = 0)

        if enable_tracking:

            wandb.log(
                        data={
                            "val/loss_fillin": val_fitb_loss if fitb else None,
                            "val/loss_caption": val_fc_loss if fc else None,
                            "val/Class_Accuracy": accuracy["Class Accuracy"] if accuracy is not None else None,
                            "val/Instance_Accuracy": accuracy["Instance Accuracy"] if accuracy is not None else None,
                            "val/Same_Accuracy": accuracy["Same Accuracy"] if accuracy is not None else None,
                            "val/Diff_Accuracy": accuracy["Diff Accuracy"] if accuracy is not None else None,
                            "val/Cider": cap_metrics["cider"] if cap_metrics is not None else None,
                            "val/Meteor": cap_metrics["meteor"] if cap_metrics is not None else None,
                            "val/Rouge": cap_metrics["rouge"] if cap_metrics is not None else None,
                        }
                    )
            
    summary_metrics ={
        "best_fitb_epoch": -1,
        "best_cid_epoch": -1,
        "best_met_epoch": -1,
        "best_score": -1,
        "best_instance_acc": -1,
        "best_same_acc": -1,
        "best_diff_acc": -1,
        "best_cider": -1,
        "best_meteor": -1,
    }


    best_fitb_model = None
    best_cid_model = None
    best_met_model = None

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        if joint:
            avg_loss_fitb, avg_loss_fc = train_model_joint(mst_model, mst_optimizer, train_dataloader, opt.grad_clip)
        elif fitb:
            avg_loss_fitb = train_model_fitb(mst_model, mst_optimizer, train_dataloader, opt.grad_clip)
        else:
            avg_loss_fc = train_model_fc(mst_model, mst_optimizer, train_dataloader, opt.grad_clip)

        if is_overfit:

            if enable_tracking: 
                wandb.log(
                        data={
                            "train/loss_fillin": avg_loss_fitb if fitb else None,
                            "train/loss_caption": avg_loss_fc if fc else None,
                            "epoch": t, 
                            "learning_rate": opt.learning_rate,
                        },
                        step = t + 1
                    )
            
            continue

        accuracy = None
        cap_metrics = None
        if fitb:
            val_fitb_loss, _, accuracy, _ = eval_model_fitb(
                            mst_model, val_dataloader, eval_kwargs=eval_kwargs
                        )
        
        if fc:
            val_fc_loss = train_eval_model_caption(mst_model, val_dataloader) 
            cap_metrics = evaluate_caption(mst_model, val_dataloader, vocab, epoch = t + 1)

        if enable_tracking:

            wandb.log(
                        data={
                            "train/loss_fillin": avg_loss_fitb if fitb else None,
                            "train/loss_caption": avg_loss_fc if fc else None,
                            "val/loss_fillin": val_fitb_loss if fitb else None,
                            "val/loss_caption": val_fc_loss if fc else None,
                            "val/Class_Accuracy": accuracy["Class Accuracy"] if accuracy is not None else None,
                            "val/Instance_Accuracy": accuracy["Instance Accuracy"] if accuracy is not None else None,
                            "val/Same_Accuracy": accuracy["Same Accuracy"] if accuracy is not None else None,
                            "val/Diff_Accuracy": accuracy["Diff Accuracy"] if accuracy is not None else None,
                            "val/Cider": cap_metrics["cider"] if cap_metrics is not None else None,
                            "val/Meteor": cap_metrics["meteor"] if cap_metrics is not None else None,
                            "val/Rouge": cap_metrics["rouge"] if cap_metrics is not None else None,
                            "val/SPICE": cap_metrics["spice"] if cap_metrics is not None else None,
                            "val/iSPICE": cap_metrics["ispice"] if cap_metrics is not None else None,
                        }
                    )

        if fitb:
            if accuracy["Class Accuracy"] > summary_metrics["best_score"]:
                summary_metrics["best_score"] = accuracy["Class Accuracy"]
                summary_metrics["best_epoch"] = t
                summary_metrics["best_diff_acc"] = accuracy["Diff Accuracy"]
                summary_metrics["best_same_acc"] = accuracy["Same Accuracy"]
                summary_metrics["best_instance_acc"] = accuracy["Instance Accuracy"]
                best_fitb_model = deepcopy(mst_model.state_dict())
        
        if fc:
            if cap_metrics["cider"] > summary_metrics["best_cider"]:
                summary_metrics["best_cider"] = cap_metrics["cider"]
                summary_metrics["best_cid_epoch"] = t
                best_cid_model = deepcopy(mst_model.state_dict())
            
            if cap_metrics["meteor"] > summary_metrics["best_meteor"]:
                summary_metrics["best_meteor"] = cap_metrics["meteor"]
                summary_metrics["best_met_epoch"] = t
                best_met_model = deepcopy(mst_model.state_dict())
       
    if enable_tracking and not is_overfit:
        wandb.log(
            data=summary_metrics
        )
    
    
    if not is_overfit:
        if fitb:
            torch.save(best_fitb_model, f"{opt.save_path}/{opt.run_name}_fitb.pth")
        
        if fc:
            torch.save(best_cid_model, f"{opt.save_path}/{opt.run_name}_cid.pth")
            torch.save(best_met_model, f"{opt.save_path}/{opt.run_name}_met.pth")
        
    print("Done!")

def setup_main():
    opt =  OmegaConf.load("config_base.yaml")
    if opt.track:
        wandb.init(project = "lsmdc-fillin-pro", name = opt.run_name)
        wandb.config.update(OmegaConf.to_container(opt, resolve=True))

    seed_everything(seed=opt.seed, track=opt.track)
    train(opt)


if __name__ == "__main__":

    setup_main()

