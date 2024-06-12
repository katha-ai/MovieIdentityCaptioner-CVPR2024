from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import pickle

from torch.utils.data import Dataset
from collections import defaultdict

import math


from ast import literal_eval


def zero_pad(features, n_feat):
    if features.shape[0] < n_feat:
        features = np.vstack(
            (features, np.zeros((n_feat - features.shape[0], features.shape[1])))
        )
    return features


class LSMDCDataset(Dataset):

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_wtoi(self):
        return self.word_to_ix

    def get_seq_length(self):
        return self.seq_length

    def get_blank_token(self):
        return self.word_to_ix["<blank>"]

    def __init__(self, opt, groups, split_ix, split_type, split_map, info):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.meta_data = {
            "max_bbox": -1,
            "max_age": -1,
            "max_cluster": -1,
            "count": 0,
        }
        self.cluster_analysis = {"cluster_nums": []}
        self.frame_analysis = {"n_frames": -1}
        self.bbox_analysis = {
            "bbox_x":[],
            "bbox_y":[],
            "over_list":[]
        }

        self.split_type = split_type

        self.input_fc_dir = os.path.join(self.opt.data_dir, self.opt.input_fc_dir)
        self.input_clip_dir = os.path.join(self.opt.data_dir,self.opt.input_clip_dir)
        self.input_arc_face_dir = os.path.join(self.opt.data_dir,self.opt.input_arc_face_dir)
        self.input_arc_face_clusters = os.path.join(self.opt.data_dir, self.opt.input_arc_face_clusters)
        self.vid_frame_width = 854
        self.vid_frame_height = 480

        with open(self.input_arc_face_clusters,"r") as f:
            self.arc_face_cluster_by_group = json.load(f)

        # New DataLoader Feature Variables
        self.mean_pool_video = True
        self.max_videos_in_group = 5


        # CLIP - Video Length limit
        self.max_vid_length = 10
        # Since we take 5 frames per second during processing
        self.max_frames = self.max_vid_length * 5
        self.vid_clip_dim = 512

        # Arc face features
        self.arc_face_dim = 512
        self.max_arc_face = 300
        self.arc_face_bbox_dim = 4

        self.max_frames_total = 900


        # load the json file which contains additional information about the dataset
        self.info = info
        self.ix_to_word = self.info["ix_to_word"]
        self.word_to_ix = self.info["word_to_ix"]
        self.groups = groups
        self.movie_dict = self.info["movie_ids"]
        self.seq_length = self.info["max_seq_length"]
        self.max_caption_length = 120
        self.vocab_size = len(self.ix_to_word)
        print("vocab size is ", self.vocab_size)
        print("max sequence length in data is", self.seq_length)

        self.h5_label_file = h5py.File(self.opt.data_dir + self.opt.input_label_h5, "r")
        self.captions = self.h5_label_file["labels"].value
        self.max_characters = self.info["max_character_count"]  # 17
        self.unique_characters = self.info["unique_character_count"]  # 11
        self.max_seg = self.opt.nsegments

        self.use_bert_embedding = self.opt.use_bert_embedding
        self.bert_size = self.opt.bert_size  # 768 * 2
        if self.use_bert_embedding:
            self.bert_embedding_dir = self.opt.data_dir + self.opt.bert_embedding_dir
            print(f"bert_embedding_dir : {self.bert_embedding_dir}")
            self.bert_embedding = {
                "train": pickle.load(
                    open(
                        os.path.join(self.bert_embedding_dir, "train_embeddings.pkl"),
                        "rb",
                    )
                ),
                "val": pickle.load(
                    open(
                        os.path.join(self.bert_embedding_dir, "val_embeddings.pkl"),
                        "rb",
                    )
                ),
                "test": pickle.load(
                    open(
                        os.path.join(self.bert_embedding_dir, "test_embeddings.pkl"),
                        "rb",
                    )
                ),
            }

        self.split_ix = split_ix
        self.split_map = split_map


    # mean pool the features across max_seg segments
    def meanpool_segments(self, features):
        if features.shape[0] >= self.max_seg:
            tmp_feat = []
            # self.max_seg - T - 5
            nps = int(
                np.floor(features.shape[0] // self.max_seg)
            )  # numbers per segment
            for i in range(self.max_seg):
                if i != self.max_seg - 1:
                    segment = features[nps * i : nps * (i + 1)]
                else:
                    segment = features[nps * i :]
                segment = segment.mean(axis=0)
                tmp_feat.append(segment)
            features = np.array(tmp_feat)
        else:
            # 0 pad frames
            features = zero_pad(features, self.max_seg)
        # features will be --> 5x1024
        return features

    def get_sent_num(self, index):
        return len(self.groups[index]["videos"])

    def get_slot_batch(self, index):
        """
        :param index:
        :return: sent_ix =  array of indices of clips for each slot (len is # of total characters in video)
                 sent_num = # of clips for the video index
        """
        v_idx = self.groups[index]["videos"]
        slots = []
        for i, id in enumerate(v_idx):
            n_blanks = self.info["videos"][id]["num_blanks"]
            if n_blanks > 0:
                for _ in range(n_blanks):
                    slots.append(i)
        return slots


    
    

    def get_character_batch(self, index):
        v_idx = self.groups[index]["videos"]
        character_ids = []
        for id in v_idx:
            n_blanks = self.info["videos"][id]["num_blanks"]
            if n_blanks > 0:
                characters = self.info["videos"][id]["character_id"]
                for n in range(n_blanks):
                    character_ids.append(characters[n])

        character_map = {}
        for c in character_ids:
            if c not in character_map:
                character_map[c] = len(character_map.keys()) + 1
        character_ids = [character_map[c] for c in character_ids]
        return character_ids

    def get_bert_batch(self, index, split):
        main_split = split
        # 'overfit': [], 'train_val'
        if split == "overfit" or split == "train_val":
            main_split = "train"

        bert_idx = self.split_map[split][self.groups[index]["id"]]
        return self.bert_embedding[main_split][bert_idx]


    def convert_to_normalized_coordinates(self,bbox_coords):
        """
        Converts bounding box coordinates to normalized coordinates.

        Parameters:
            bbox_coords (np.ndarray): Array of bounding box coordinates (xmin, ymin, xmax, ymax).
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            np.ndarray: Array of normalized coordinates (xcenter/img_width, ycenter/img_height,
                        width/img_width, height/img_height).
        """
        x_min, y_min, x_max, y_max = bbox_coords

        if x_max > self.vid_frame_width:
            x_min = x_min * self.vid_frame_width/x_max
            x_max = self.vid_frame_width

        if y_max > self.vid_frame_height:
            y_min = y_min * self.vid_frame_height/y_max
            y_max = self.vid_frame_height


        norm_x_min = x_min/self.vid_frame_width
        norm_x_max = x_max/self.vid_frame_width
        norm_y_min = y_min/self.vid_frame_height
        norm_y_max = y_max/self.vid_frame_height

        normalized_coords = np.array([norm_x_min, norm_y_min, norm_x_max, norm_y_max])
        return normalized_coords

      
    def get_nba_gt(self,index):
        group = self.groups[index]

        
        nba_caption =  group["nba_caption"]
        nba_caption_mask = group["nba_caption_mask"]
        nba_segment_ids = group["nba_segment_ids"]
        nba_position_ids = group["nba_position_ids"]
        nba_blank_masks = group["nba_blank_mask"]
        nba_blank_indexes = group["nba_blank_indexes"]

        return nba_caption, nba_caption_mask, nba_segment_ids, nba_position_ids, nba_blank_masks, nba_blank_indexes
        


    def get_seg_batch(self, index):
        """
        :param index: group index in batch
        :return: fc_features   [sent_num x D_fc]
                 face_features [sent_num x face_num x D_ff]
        """
        ovr_index = self.groups[index]["id"]
        v_idx = self.groups[index]["videos"]
        sent_num = len(v_idx)
        assert sent_num > 0, "data should have at least one caption"
        fc_features = []

        video_clip_features = []
        video_masks = []
        video_segments = []

        # Arc face features
        arc_face_features = np.empty([0, 512], dtype=float)
        arc_face_masks = np.zeros(self.max_arc_face, dtype="float32")
        arc_face_segments = []
        arc_face_vid_ids = []
        arc_face_frame_embeddings = np.empty([0, 512], dtype=float)
        

        arc_face_bbox = np.empty([0, 4], dtype=float)
        arc_face_gender = []
        arc_face_age = []

        # BERT features
        bert_batch = np.zeros((self.max_characters, self.bert_size), dtype="float32")

        if self.use_bert_embedding:
            bert_batch = self.get_bert_batch(index, self.split_type)

        # # Caption features
        nba_caption, nba_caption_mask, nba_segment_ids, nba_position_ids, nba_blank_masks, nba_blank_indexes = self.get_nba_gt(index)
        

        slots = self.get_slot_batch(index)
        slot_num = len(slots)
        sent_num = self.get_sent_num(index)

        vids_in_group = len(self.groups[index]["videos"])

        # Character info
        character_ids = []
        genders = []
        if(self.split_type != "test"):
            character_ids = self.get_character_batch(index)
        infos = []

        for id in v_idx:
            info_dict = {}
            info_dict["index"] = id
            info_dict["g_index"] = index
            info_dict["id"] = self.info["videos"][id]["clip"]
            info_dict["caption"] = self.captions[id]
            info_dict["skipped"] = slot_num == 0
            infos.append(info_dict)

            movie = self.info["videos"][id]["movie"]
            clip = self.info["videos"][id]["clip"]

            video_clip_dir = [self.input_clip_dir, movie, clip + ".npy"]

            video_clip_feats = np.load(os.path.join(*video_clip_dir))

            video_mask = np.zeros(self.max_frames, dtype="float32")
            vid_segments = np.zeros(self.max_frames, dtype="float32")

            # Limit the input to max_frames
            n_frames_actual = video_clip_feats.shape[0]
            video_clip_feats = video_clip_feats[: self.max_frames]

            if video_clip_feats.shape[0] > 0:
                video_clip_feat_data = zero_pad(video_clip_feats, self.max_frames)
                video_mask[: min(video_clip_feats.shape[0], self.max_frames)] = 1

            # Getting video_clip segments
            n_frames = video_clip_feats.shape[0]
            segments = np.arange(0, n_frames) / (n_frames / 5)
            vid_segments_list = list(map(math.floor, segments))
            vid_segments[
                : min(self.max_frames, len(vid_segments_list))
            ] = vid_segments_list

            
            arc_face_vid_dir = [
                self.input_arc_face_dir,
                movie,
                clip + "_concatenated.npy",
            ]
            arc_face_vid_path = os.path.join(*arc_face_vid_dir)
            if os.path.isfile(arc_face_vid_path):
                arc_face_feat = np.load(arc_face_vid_path)
                arc_face_features = np.concatenate((arc_face_features, arc_face_feat))

                arc_face_vid_ids.extend([v_idx.index(id)] * arc_face_feat.shape[0])

                # getting meta data
                info_dir = [self.input_arc_face_dir, movie, clip + ".info"]
                with open(os.path.join(*info_dir), "r") as f:
                    info_data = json.load(f)
                arc_face_frame_nums = [i["frame_num"] for i in info_data]
                total_frames = (
                    n_frames_actual
                ) * 5  # As the videos are sampled at every 5th frame
                arc_face_seg = [5 * (i / total_frames) for i in arc_face_frame_nums]

                arc_face_segments_list = list(map(math.floor, arc_face_seg))
                arc_face_segments.extend(arc_face_segments_list)



                # for age and gender features
                arc_face_age.extend([int(i["age"]) for i in info_data])
                arc_face_gender.extend([int(i["gender"]) for i in info_data])

                
                # for bbox features
                for i in info_data:
                    arc_face_bbox = np.concatenate(
                        (
                            arc_face_bbox,
                            np.expand_dims(
                                self.convert_to_normalized_coordinates(np.absolute(np.array(literal_eval(i["bbox"])))), axis=0
                            ),
                        )
                    )

            
            


            fc_dir = [self.input_fc_dir, movie, clip + ".npy"]
            fc_feats = np.load(os.path.join(*fc_dir))
            if self.mean_pool_video:
                fc_feats = self.meanpool_segments(fc_feats)

            
            fc_features.append(fc_feats) # Shape would be 5x1024 now.


            video_clip_features.append(video_clip_feat_data)
            video_masks.append(video_mask)
            video_segments.append(vid_segments)
            

        # arc face features processing

        arc_face_features = arc_face_features[: self.max_arc_face]
        arc_face_segments = arc_face_segments[: self.max_arc_face]
        arc_face_vid_ids = arc_face_vid_ids[: self.max_arc_face]
        arc_face_bbox = arc_face_bbox[: self.max_arc_face]

        arc_face_gender = arc_face_gender[: self.max_arc_face]
        arc_face_age = arc_face_age[: self.max_arc_face]
        arc_face_frame_embeddings = arc_face_frame_embeddings[: self.max_arc_face]
        arc_face_features_final = []
        arc_face_bbox_final = []
        arc_face_cluster_offline = []
        if arc_face_features.shape[0] > 0:
            arc_face_features_final = zero_pad(arc_face_features, self.max_arc_face)
            arc_face_masks[: min(arc_face_features.shape[0], self.max_arc_face)] = 1
            arc_face_segments.extend(
                [0] * (self.max_arc_face - arc_face_features.shape[0])
            )
            arc_face_vid_ids.extend(
                [0] * (self.max_arc_face - arc_face_features.shape[0])
            )
            arc_face_gender.extend(
                [0] * (self.max_arc_face - arc_face_features.shape[0])
            )
            arc_face_age.extend([0] * (self.max_arc_face - arc_face_features.shape[0]))

            arc_face_cluster_offline = self.arc_face_cluster_by_group[str(ovr_index)][:]
            arc_face_cluster_offline.extend(
                [0] * (self.max_arc_face - arc_face_features.shape[0])
            )

            arc_face_bbox_final = zero_pad(arc_face_bbox, self.max_arc_face)
        

        
        
        
        return (
            np.array(nba_caption), 
            np.array(nba_caption_mask), 
            np.array(nba_segment_ids), 
            np.array(nba_position_ids), 
            np.array(nba_blank_masks), 
            np.array(nba_blank_indexes),
            np.array(fc_features),
            np.array(video_clip_features),
            np.array(video_masks),
            np.array(video_segments),
            np.array(arc_face_features_final),
            np.array(arc_face_masks),
            np.array(arc_face_segments),
            np.array(arc_face_vid_ids),
            np.array(arc_face_cluster_offline),
            np.array(arc_face_bbox_final),
            np.array(arc_face_gender),
            np.array(arc_face_age),
            np.array(bert_batch),
            np.array(character_ids),
            np.array(genders),
            np.array(slots),
            np.array(sent_num),
            vids_in_group,
            infos
        )

 
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn"""

        return self.get_seg_batch(index), index

    def __len__(self):
        return len(self.groups)
