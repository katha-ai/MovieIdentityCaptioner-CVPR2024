from trainer import eval_model_fitb, evaluate_caption
import wandb

def fitb_ckpt(model, loader, eval_kwargs, track):
    val_loss, predictions, accuracy,gender_accuracy = eval_model_fitb(
                        model, loader, eval_kwargs=eval_kwargs
                    )
    
    if track:
            wandb.log(
                        data={
                            "val/loss_fillin": val_loss,
                            "val/Class_Accuracy": accuracy["Class Accuracy"],
                            "val/Instance_Accuracy": accuracy["Instance Accuracy"],
                            "val/Same_Accuracy": accuracy["Same Accuracy"],
                            "val/Diff_Accuracy": accuracy["Diff Accuracy"],
                        },
                        step = 0
                    )
    
    return

def fc_ckpt (model, loader, vocab, track):
    cap_metrics = evaluate_caption(model, loader, vocab, epoch = 2)

    print(cap_metrics)

    if track:
         wandb.log(
                    data={
                        "val/METEOR": cap_metrics["meteor"],
                        "val/CIDEr": cap_metrics["cider"],
                        "val/ROUGE": cap_metrics["rouge"],
                        "val/SPICE": cap_metrics["spice"],
                        "val/iSPICE": cap_metrics["ispice"],
                    },
                    step=0
                )
    
    return
      

