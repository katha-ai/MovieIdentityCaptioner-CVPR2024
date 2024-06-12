from data.datasets import LSMDCDataset
import opts
from torch.utils.data import DataLoader
import numpy as np


# Default parameters
class LSMDCDataloader():

    def __init__(self, dataset, batch_size=16, shuffle = True):

        self.dataset = dataset
        self.loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=9,
                          collate_fn=self.custom_collate)

    def custom_collate(self, data):
        batch_size = len(data)


        slot_batch = np.ones((batch_size, self.dataset.max_characters), dtype="int") * -1
        sent_num_batch = np.zeros(batch_size, dtype="int")
        video_clip_batch = np.zeros(
            [batch_size, self.dataset.max_videos_in_group, self.dataset.max_frames, self.dataset.vid_clip_dim],
            dtype="float32",
        )
        video_mask_batch = np.zeros(
            [batch_size, self.dataset.max_videos_in_group, self.dataset.max_frames], dtype="float32"
        )
        video_segment_batch = np.zeros(
            (batch_size, self.dataset.max_videos_in_group, self.dataset.max_frames), dtype="int"
        )

        # arc face features
        arc_face_batch = np.zeros(
            [batch_size, self.dataset.max_arc_face, self.dataset.arc_face_dim],
            dtype="float32",
        )
        arc_face_mask_batch = np.zeros(
            [batch_size, self.dataset.max_arc_face],
            dtype="float32",
        )
        arc_face_segment_batch = np.zeros(
            [batch_size, self.dataset.max_arc_face],
            dtype="float32",
        )
        # to do
        arc_face_vid_id_batch = np.zeros(
            [batch_size, self.dataset.max_arc_face],
            dtype="float32",
        )
        arc_face_cluster_batch = np.zeros(
            [batch_size, self.dataset.max_arc_face],
            dtype="float32",
        )
        arc_face_bbox_batch = np.zeros(
            [batch_size, self.dataset.max_arc_face, self.dataset.arc_face_bbox_dim],
            dtype="float32",
        )
        arc_face_gender_batch = np.zeros([batch_size, self.dataset.max_arc_face], dtype="int")
        arc_face_age_batch = np.zeros([batch_size, self.dataset.max_arc_face], dtype="int")

        
        fc_batch = np.zeros(
            [batch_size, self.dataset.max_videos_in_group, self.dataset.max_seg, self.dataset.opt.fc_feat_size],
            dtype="float32",
        )

        
        bert_batched = np.zeros(
            (batch_size, self.dataset.max_characters, self.dataset.bert_size), dtype="float32"
        )

        
        # NBA captions
        nba_caption_batched = np.zeros((batch_size, self.dataset.max_caption_length), dtype="int")
        nba_mask_batched = np.zeros((batch_size, self.dataset.max_caption_length), dtype="float32")

        nba_caption_position_batch = np.zeros(
            (batch_size, self.dataset.max_caption_length), dtype="int"
        )
        nba_caption_segment_batch = np.zeros(
            (batch_size, self.dataset.max_caption_length), dtype="int"
        )

        nba_blank_masks_batch = np.zeros((batch_size, self.dataset.max_caption_length), dtype="int")

        nba_blank_indexes_batch = np.zeros((batch_size, self.dataset.max_characters), dtype="int")


        gender_batch = (None)
        character_batch = np.zeros((batch_size, self.dataset.max_characters + 1), dtype="int")
        slot_mask_batch = np.zeros((batch_size, self.dataset.max_characters + 1), dtype="int")


        infos_batch = []
        slot_size = []

        for i, items in enumerate(data):
            (  
                nba_caption, 
                nba_caption_mask, 
                nba_segment_ids, 
                nba_position_ids, 
                nba_blank_masks, 
                nba_blank_indexes,
                fc_features,
                video_clip_features,
                video_masks,
                video_segments,
                arc_face_features_final,
                arc_face_masks,
                arc_face_segments,
                arc_face_vid_ids,
                arc_face_cluster_offline,
                arc_face_bbox_final,
                arc_face_gender,
                arc_face_age,
                bert_batch,
                character_ids,
                genders,
                slots,
                sent_num,
                vids_in_group,
                infos
            ) = items[0]

            slot_num = len(slots)
            slot_size.append(slot_num)
            infos_batch.extend(infos)

            if(len(character_ids) != 0):
                character_batch[i, 1 : slot_num + 1] = character_ids
                            

            fc_batch[i, :vids_in_group, :, :] = fc_features
            video_clip_batch[i, :vids_in_group, :, :] = video_clip_features
            video_mask_batch[i, :vids_in_group, :] = video_masks
            video_segment_batch[i, :vids_in_group, :]  = video_segments

            if arc_face_features_final.shape[0] > 0:
                arc_face_batch[i] = arc_face_features_final
                arc_face_mask_batch[i] = arc_face_masks
                arc_face_segment_batch[i] = arc_face_segments
                arc_face_vid_id_batch[i] = arc_face_vid_ids
                arc_face_cluster_batch[i] = arc_face_cluster_offline
                arc_face_bbox_batch[i] = arc_face_bbox_final
                arc_face_gender_batch[i] = arc_face_gender
                arc_face_age_batch[i] = arc_face_age
            


            if self.dataset.use_bert_embedding:
                bert_batched[i] = bert_batch
            
            nba_caption_batched[i] = nba_caption
            nba_mask_batched[i] = nba_caption_mask
            nba_caption_segment_batch[i] = nba_segment_ids
            nba_caption_position_batch[i] = nba_position_ids
            nba_blank_masks_batch[i] = nba_blank_masks
            nba_blank_indexes_batch[i] = nba_blank_indexes
            

            sent_num_batch[i] = sent_num
            slot_batch[i, :slot_num] = slots
            slot_mask_batch[i, : slot_num + 1] = 1

        data = {}

        data["arc_face"] = {
            "feats": arc_face_batch,
            "masks": arc_face_mask_batch,
            "segments": arc_face_segment_batch,
            "vid_ids": arc_face_vid_id_batch,
            "clusters": arc_face_cluster_batch,
            "bbox": arc_face_bbox_batch
        }
        data["video_clip"] = {
            "feats": video_clip_batch,
            "masks": video_mask_batch,
            "segments": video_segment_batch
        }
        data["fc"] = {
            "feats": fc_batch
        }


        data["nba"] = {
            "position_ids": nba_caption_position_batch,
            "segment_ids": nba_caption_segment_batch,
            "gt_captions": nba_caption_batched,
            "gt_masks": nba_mask_batched,
            "blank_masks": nba_blank_masks_batch,
            "blank_indexes": nba_blank_indexes_batch,
            "bert_emb": bert_batched
        }

        data["slot"] = {
            "slots": slot_batch,
            "slot_masks": slot_mask_batch
        }

        data["slot_sent"] = {
            "sent_num": sent_num_batch,
            "slot_size": np.max(slot_size)
        }

        data["char_gen"] = {
            "characters":character_batch,
            "genders": gender_batch
        }

        data["infos"] = infos_batch

        return data
    
    def get_loader(self):
        return self.loader