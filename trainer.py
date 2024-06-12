import csv
import json
import os
import random
import string
import subprocess
import torch
import misc.utils as utils
from tqdm import tqdm
import numpy as np
import re

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from iSPICE.ispice import Spice

def capitalize_person_ids(text):
    # Use regular expression to find and replace person IDs
    capitalized_text = re.sub(r'\bp(\d+)\b', lambda match: f'P{match.group(1)}', text)
    return capitalized_text

id_tokens = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 
             'P11', 'P12','P13', 
             'P14', 'P15',
             'P16', 'P17','P18']


def order_ids_2(text):
    text += " "
    sentences = text.split('. ')
    capitalized_sentences = []
    token_map = {}
    current_token_index = 0

    for sentence in sentences:
        words = sentence.split()
        if words:
            for i, word in enumerate(words):
                if word.upper() in id_tokens:
                    if word not in token_map:
                        token_map[word] = id_tokens[current_token_index]
                        current_token_index += 1
                    words[i] = token_map[word]
            capitalized_sentences.append(" ".join(words))
            
    capitalized_text = '. '.join(capitalized_sentences)
    return capitalized_text

def capitalize_first_word_only(text):
    text += " "
    sentences = text.split('. ')
    capitalized_sentences = []

    for sentence in sentences:
        words = sentence.split()
        if words:
            first_word = words[0].capitalize()
            rest_of_sentence = ' '.join(words[1:])
            capitalized_sentence = f'{first_word} {rest_of_sentence}'
            capitalized_sentences.append(capitalized_sentence)

    capitalized_text = '. '.join(capitalized_sentences)
    return capitalized_text



def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def evaluate_caption(model, data_loader, vocab, epoch):

    model.eval()
    
    meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    ispice_scorer = Spice()

    all_preds = []
    all_gt = []

    rouge_list = []
    meteor_list = []
    cider_list =[]

    for _, data in tqdm(enumerate(data_loader)):

        tmp = [
                data["nba"],
                data["arc_face"],
                data["video_clip"],
                data["fc"],
                data["slot"],
            ]
        

        for feat_group in tmp:
            for key in feat_group.keys():
                feat_group[key] = feat_group[key] if feat_group[key] is None else torch.from_numpy(feat_group[key]).cuda()
        
        

        (   

            nba,
            arc_face,
            video_clip,
            fc,
            slot,
        ) = tmp
        


        with torch.no_grad():

            
            predicted_captions = model(
                dict(nba),
                dict(arc_face),
                dict(video_clip),
                dict(fc),
                dict(slot),
                dict(data["slot_sent"]),
                mode="predict_caption"
            )
            gt_tmp = dict(nba)


            gt_tmp["gt_captions"][:,0] = 0
            gt_tmp["gt_captions"][gt_tmp["gt_captions"] == 30537] = 0
            gt_tmp["gt_captions"][gt_tmp["gt_captions"] == 30523] = 0
            gt_tmp["gt_captions"][gt_tmp["gt_captions"] == 102] = 1012 # converting [SEP] tokens to "."

            
            predicted_captions[predicted_captions == 30523] = 0
            predicted_captions[predicted_captions == 30536] = 0
            predicted_captions[predicted_captions == 30537] = 0
            predicted_captions[predicted_captions == 102] = 1012


            predictions = vocab.batch_decode(predicted_captions,skip_special_tokens = True)
            ground_truth = vocab.batch_decode(gt_tmp["gt_captions"], skip_special_tokens = True)

            all_preds.extend(predictions)
            all_gt.extend(ground_truth)


            hypotheses = {'image'+str(i): [capitalize_first_word_only(capitalize_person_ids(order_ids_2(predictions[i])))] for i in range(len(predictions))}
            references = {'image'+str(i): [capitalize_first_word_only(capitalize_person_ids(order_ids_2(ground_truth[i])))] for i in range(len(ground_truth))}


            _, ml = meteor_scorer.compute_score(references, hypotheses)
            _, rl = rouge_scorer.compute_score(references, hypotheses)
            _, cl = cider_scorer.compute_score(references, hypotheses)

            
            meteor_list.extend(ml)
            rouge_list.extend(rl)
            cider_list.extend(cl)

    if epoch > 1:

        hypotheses = {'image'+str(i): [capitalize_first_word_only(capitalize_person_ids(order_ids_2(all_preds[i])))] for i in range(len(all_preds))}
        references = {'image'+str(i): [capitalize_first_word_only(capitalize_person_ids(order_ids_2(all_gt[i])))] for i in range(len(all_gt))}

        spice_score,_,ispice_score,_ = ispice_scorer.compute_score(references, hypotheses)
    
    else:
        spice_score = 0
        ispice_score = 0
            
   
    caption_metrics = {
        "meteor": np.mean(meteor_list),
        "rouge":np.mean(rouge_list),
        "cider":np.mean(cider_list),
        "spice": spice_score,
        "ispice": ispice_score
    }

    return caption_metrics


def train_model_joint(model, optimizer, data_loader, grad_clip=0.1):
    model.train()

    losses_fillin = []
    losses_caption = []
    
    for batch_idx, data in tqdm(enumerate(data_loader)):


        tmp = [ 
                data["nba"],
                data["arc_face"],
                data["video_clip"],
                data["fc"],
                data["slot"],
            ]
        

        for feat_group in tmp:
            for key in feat_group.keys():
                feat_group[key] = feat_group[key] if feat_group[key] is None else torch.from_numpy(feat_group[key]).cuda()
        
        

        (   
            nba,
            arc_face,
            video_clip,
            fc,
            slot,
        ) = tmp
        

        optimizer.zero_grad()

        loss_fillin = model(
            dict(nba),
            dict(arc_face),
            dict(video_clip),
            dict(fc),
            dict(slot),
            dict(data["slot_sent"]),
            mode="forward_fitb"
        )

        
        

        loss_fillin = loss_fillin.mean()

        print(f"Fillin Loss value is : {loss_fillin}")

        loss_fillin.backward()
        
        utils.clip_gradient(optimizer, grad_clip)
        optimizer.step()
        torch.cuda.synchronize()

        
        loss_caption = model(
            dict(nba),
            dict(arc_face),
            dict(video_clip),
            dict(fc),
            dict(slot),
            dict(data["slot_sent"]),
            mode="forward_caption"
        )

        
        
        optimizer.zero_grad()

        loss_caption = loss_caption.mean()

        print(f"Caption Loss value is : {loss_caption}")

        loss_caption.backward()
        
        utils.clip_gradient(optimizer, grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        
        
        losses_fillin.append(loss_fillin.detach().item())
        losses_caption.append(loss_caption.detach().item())
        
        
    avg_gen_loss_fillin = np.mean(losses_fillin)
    avg_gen_loss_caption = np.mean(losses_caption)


    return avg_gen_loss_fillin, avg_gen_loss_caption


def train_eval_model_caption(model, data_loader):
    model.eval()
    
    total_batches = len(data_loader)
    losses = []
    
    for batch_idx, data in tqdm(enumerate(data_loader)):
        tmp = [
                data["nba"],
                data["arc_face"],
                data["video_clip"],
                data["fc"],
                data["slot"],

            ]
        

        for feat_group in tmp:
            for key in feat_group.keys():
                feat_group[key] = feat_group[key] if feat_group[key] is None else torch.from_numpy(feat_group[key]).cuda()
        
        

        (
            nba,
            arc_face,
            video_clip,
            fc,
            slot,
        ) = tmp
        
        with torch.no_grad():

            
            loss = model(
                dict(nba),
                dict(arc_face),
                dict(video_clip),
                dict(fc),
                dict(slot),
                dict(data["slot_sent"]),
                mode="forward_caption"
            )

            
        

        loss = loss.mean()

        print(f"Loss value is : {loss}")
        losses.append(loss.item())
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}")
    
    avg_gen_loss = np.mean(losses)

    return avg_gen_loss

def train_model_fitb(model, optimizer, data_loader, grad_clip=0.1):
    model.train()
    
    total_batches = len(data_loader)
    losses = []
    
    for batch_idx, data in tqdm(enumerate(data_loader)):
        tmp = [ 
                data["nba"],
                data["arc_face"],
                data["video_clip"],
                data["fc"],
                data["slot"],
            ]
        

        for feat_group in tmp:
            for key in feat_group.keys():
                feat_group[key] = feat_group[key] if feat_group[key] is None else torch.from_numpy(feat_group[key]).cuda()
        
        

        (   
            nba,
            arc_face,
            video_clip,
            fc,
            slot,
        ) = tmp
        

        optimizer.zero_grad()

        

        loss = model(
            dict(nba),
            dict(arc_face),
            dict(video_clip),
            dict(fc),
            dict(slot),
            dict(data["slot_sent"]),
            mode="forward_fitb"
        )
        

        loss = loss.mean()

        print(f"Loss value is : {loss}")

        loss.backward()
        
        utils.clip_gradient(optimizer, grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        
        losses.append(loss.item())
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}")
    
    avg_gen_loss = np.mean(losses)


    return avg_gen_loss

def train_model_fc(model, optimizer, data_loader, grad_clip=0.1):
    model.train()
    
    total_batches = len(data_loader)
    losses = []
    
    for batch_idx, data in tqdm(enumerate(data_loader)):
        tmp = [ 
                data["nba"],
                data["arc_face"],
                data["video_clip"],
                data["fc"],
                data["slot"],
            ]
        

        for feat_group in tmp:
            for key in feat_group.keys():
                feat_group[key] = feat_group[key] if feat_group[key] is None else torch.from_numpy(feat_group[key]).cuda()
        
        

        (   
            nba,
            arc_face,
            video_clip,
            fc,
            slot,
        ) = tmp
        

        optimizer.zero_grad()

        

        loss = model(
            dict(nba),
            dict(arc_face),
            dict(video_clip),
            dict(fc),
            dict(slot),
            dict(data["slot_sent"]),
            mode="forward_caption"
        )
        

        loss = loss.mean()

        print(f"Loss value is : {loss}")

        loss.backward()
        
        utils.clip_gradient(optimizer, grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        
        losses.append(loss.item())
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}")
    
    avg_gen_loss = np.mean(losses)


    return avg_gen_loss


def eval_fillin(preds, model_id, split, remove=False):
    import sys

    sys.path.append("codalab-2019-fill_in")
    results = []
    for pred in preds:
        info = {"video_id": pred["video_id"]}
        info["characters"] = pred["characters"]
        results.append(info)
    if remove:
        model_id += id_generator()  # to avoid processing and removing same ids
    with open(
        os.path.join("character_eval", "characters", "character_" + model_id + ".csv"),
        "w",
    ) as f:
        keys = ["video_id", "characters"]
        dict_writer = csv.DictWriter(f, keys, delimiter="\t")
        dict_writer.writerows(results)
        f.close()

    eval_command = [
        "python",
        "eval_characters.py",
        "-s",
        "characters/character_" + model_id + ".csv",
        "--split",
        split,
        "-o",
        "results/result_" + model_id + ".json",
    ]
    subprocess.call(eval_command, cwd="character_eval")

    with open(
        os.path.join("character_eval", "results", "result_" + model_id + ".json"), "r"
    ) as f:
        output = json.load(f)
        f.close()
    if remove:  # remove for validation
        os.remove(
            os.path.join(
                "character_eval", "characters", "character_" + model_id + ".csv"
            )
        )
        os.remove(
            os.path.join("character_eval", "results", "result_" + model_id + ".json")
        )
    return output


def calculate_gender_accuracy(preds):
    correct = []
    f_recall = []
    f_precision = []
    for pred in preds:
        if type(pred["genders"]) is not str:
            for i in range(len(pred["genders"])):
                correct.append(pred["genders"][i] == pred["gt_genders"][i])
                if pred["gt_genders"][i] == 1:
                    f_recall.append(pred["genders"][i] == pred["gt_genders"][i])
                if pred["genders"][i] == 1:
                    f_precision.append(pred["genders"][i] == pred["gt_genders"][i])

    return np.mean(correct), np.mean(f_recall), np.mean(f_precision)


def decode_sequence(ix_to_word, seq):
    D = seq.shape[0]
    txt = ""
    for j in range(D):
        ix = seq[j]
        if ix > 0:
            if j >= 1:
                txt = txt + " "
            txt = txt + ix_to_word[str(ix)]
        else:
            break
    return txt

def lst2string(lst):
    txt = ""
    for l in lst:
        txt += "[" + str(l) + "]" + ","
    return txt[:-1]

def get_predictions(data, split, vocab_size, characters, predicted_characters, genders, predicted_genders, slots, eval_accuracy=1, verbose=True):
    g_id = -1
    b = -1
    s = 0
    b_s = 0
    alphas = None
    predictions = []
    slots = slots.data.cpu().numpy()
    # print and store actual decoded sentence
    for info in data["infos"]:
        entry = {
            "video_id": info["id"],
            "group_id": info["g_index"],
            "caption": decode_sequence(vocab_size, info["caption"][1:-1]),
            "characters": "_",
            "genders": "_",
            "gt_characters": "_",
            "gt_genders": "_",
        }
        if g_id != entry["group_id"]:
            if not info["skipped"]:
                b += 1
            s_k = 0
            b_s = 0
            g_id = entry["group_id"]

        if not info["skipped"] and s_k in slots[b] and eval_accuracy:
            # ipdb.set_trace()
            num_clips = np.count_nonzero(slots[b] == s_k)
            entry["characters"] = lst2string(
                predicted_characters[s : s + num_clips]
            )
            entry["genders"] = predicted_genders[s : s + num_clips]

            if alphas is not None:
                entry["alphas"] = alphas[b]  # , s_k]

            if split != "test":
                entry["gt_characters"] = lst2string(
                    characters[b][b_s : b_s + num_clips].data.cpu().numpy()
                )
                if genders is not None:
                    classify_gender = True
                    entry["gt_genders"] = (
                        genders[b][b_s : b_s + num_clips].data.cpu().numpy()
                    )

            s += num_clips
            b_s += num_clips

        s_k += 1
        predictions.append(entry)

        if verbose:
            print(
                "video %s: caption: %s; predicted characters: %s ; gt_characters: %s; predicted_genders: %s; gt_genders: %s"
                % (
                    entry["video_id"],
                    entry["caption"].encode("ascii", "ignore"),
                    entry["characters"],
                    entry["gt_characters"],
                    entry["genders"],
                    entry["gt_genders"],
                )
            )

        

    return predictions


def eval_model_fitb(model, data_loader, eval_kwargs={}):
   
    split = eval_kwargs.get("split", "val")
    eval_accuracy = eval_kwargs.get("eval_accuracy", 0)  # Default is 1

    remove_result = eval_kwargs.get(
        "remove", 0
    )  
    
    model_id = eval_kwargs.get("id", eval_kwargs.get("val_id", ""))


    if split == "val":
        model_id = "val_" + model_id

    model.eval()

    losses = []
    predictions = []
    classify_gender = False
    vocab_size = data_loader.dataset.get_vocab()
    total_sent = 0

    for batch_idx, data in tqdm(enumerate(data_loader)):
        tmp = [
            data["nba"],
            data["arc_face"],
            data["video_clip"],
            data["fc"],
            data["slot"],
            data["char_gen"],
        ]
        


        for feat_group in tmp:
            for key in feat_group.keys():
                feat_group[key] = feat_group[key] if feat_group[key] is None else torch.from_numpy(feat_group[key]).cuda()
        
        (   
            nba,
            arc_face,
            video_clip,
            fc,
            slot,
            char_gen,
        ) = tmp
        
        sent_num = data["slot_sent"]["sent_num"]
        total_sent += np.sum(sent_num)


        with torch.no_grad():
            # calculate loss
            if split != "test":
                

                loss = model(
                    dict(nba),
                    dict(arc_face),
                    dict(video_clip),
                    dict(fc),
                    dict(slot),
                    dict(data["slot_sent"]),
                    mode = "forward_fitb"
                )
                



                loss = loss.mean()
                losses.append(loss.item())
            if split == "test" or eval_accuracy:
            
                predicted_characters, predicted_genders = model(
                    dict(nba),
                    dict(arc_face),
                    dict(video_clip),
                    dict(fc),
                    dict(slot),
                    dict(data["slot_sent"]),
                    mode="predict_fitb",
                )
        

                predicted_characters = (
                    predicted_characters.data.cpu().numpy().astype("int")
                )
                predicted_genders = predicted_genders.data.cpu().numpy().astype("int")

        
        preds = get_predictions(data, split, vocab_size, char_gen["characters"], predicted_characters, char_gen["genders"], predicted_genders, slot["slots"])
        predictions.extend(preds)
    

    gen_loss = np.mean(losses)
    

    if eval_accuracy:
        accuracy = eval_fillin(predictions, model_id, split, remove=remove_result)
        gender_accuracy = None
        if split != "test" and classify_gender:
            (
                gender_accuracy,
                female_recall,
                female_precision,
            ) = calculate_gender_accuracy(predictions)
            print("gender_accuracy: ", gender_accuracy)
            print("female recall: ", female_recall)
            print("female precision: ", female_precision)
            print(
                "female F1: ",
                2
                * female_recall
                * female_precision
                / (female_recall + female_precision),
            )
        
    return gen_loss, predictions, accuracy, gender_accuracy