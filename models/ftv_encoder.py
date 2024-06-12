import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import copy


class FTV_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # load config parameters
        self.config = config
        self.nvid = config.nvid  # 5
        self.model_dim = config.encoding_size
        self.nhead = 8  
        self.hidden_layers_dim = 2048
        self.dropout_p = 0.1  
        self.batch_size = config.batch_size 
        self.nsegments = config.nsegments  # 5

        # Token Embedding - Special tokens, video_id, segment_id embeddings
        self.spcl_embed = nn.Embedding(self.nvid, self.model_dim)
        self.video_embed = nn.Embedding(self.nvid, self.model_dim)
        self.segment_embed = nn.Embedding(self.nsegments, self.model_dim).cuda()

        if config.run_type == "fc":
            self.type_embed = nn.Embedding(3, self.model_dim).cuda()
        else:
            self.type_embed = nn.Embedding(4, self.model_dim).cuda()

        self.clip_encode = nn.Linear(512,self.model_dim).cuda()


        # Arc face cluster embeddings & bbox embeddings
    
        if config.run_type == "fitb" or config.run_type == "joint":
            self.arc_face_encoding = nn.Linear(512, self.model_dim).cuda()

        if config.run_type == "fitb":
            self.arc_cluster_embed = nn.Embedding(300, self.model_dim).cuda()
        else:
            self.arc_cluster_embed = nn.Embedding(400, self.model_dim).cuda()
        self.arc_bbox_encode = nn.Linear(4,self.model_dim).cuda()

        if config.run_type == "fc":
            self.arc_face_encoding = nn.Linear(512, self.model_dim).cuda()


        self.encoder_layers = TransformerEncoderLayer(
            self.model_dim, self.nhead, self.hidden_layers_dim, self.dropout_p
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, 2)
        self.videoidx = torch.tensor(range(self.nvid)).to(device="cuda")
        self.frameidx = torch.tensor(range(self.nsegments)).to(device="cuda")



    def forward(
        self,
        arc_face,
        video_clip,
        fc,
        text_embeddings,
        slot,
        slot_sent,
        video_embed,
    ):
        
        self.batch_size = arc_face["feats"].shape[0]
        
        video_slots = copy.deepcopy(slot["slots"])
        video_slots[video_slots == -1] = 0
        # remove extra dimension in slot_masks
        text_masks = slot["slot_masks"][:, 1:]

        
        if text_embeddings is not None:
            text_video_ids = video_embed(video_slots)
            text_type = torch.full_like(video_slots, 0)
            text_type_embedding = self.type_embed(text_type)
            text_embeddings = text_embeddings + text_video_ids + text_type_embedding
            
        
        video_clip["segments"] = video_clip["segments"].reshape(
            self.batch_size, video_clip["segments"].shape[1] * video_clip["segments"].shape[2]
        )
        video_clip_e_t = self.segment_embed(video_clip["segments"])
        video_clip_ids = torch.tensor([i for i in range(5) for j in range(50)]).repeat(
            self.batch_size, 1
        )
        video_clip_e_i = video_embed(video_clip_ids.to(device="cuda"))
        video_clip_type = torch.full_like(video_clip_ids, 3).to(device="cuda")
        video_clip_type_embeddings = self.type_embed(video_clip_type)

        video_clip_feats = self.clip_encode(video_clip["feats"])

        video_clip_embeddings = (
            video_clip_feats
            + video_clip_e_i
            + video_clip_e_t
            + video_clip_type_embeddings
        )
        video_clip["masks"] = video_clip["masks"].reshape(
            self.batch_size, video_clip["masks"].shape[1] * video_clip["masks"].shape[2]
        )


        arc_face["segments"] = arc_face["segments"].to(torch.long)
        arc_face["vid_ids"] = arc_face["vid_ids"].to(torch.long)
        arc_face["clusters"] = arc_face["clusters"].to(torch.long)
        arc_face_e_t = self.segment_embed(arc_face["segments"])
        arc_face_e_i = video_embed(arc_face["vid_ids"])
        arc_face_type = torch.full_like(arc_face["vid_ids"], 2).to(device="cuda")
        arc_face_type_embeddings = self.type_embed(arc_face_type)

        arc_face_feats = self.arc_face_encoding(arc_face["feats"])

        arc_face_cluster_embeddings = self.arc_cluster_embed(arc_face["clusters"])
        arc_face_bbox_embeddings = self.arc_bbox_encode(arc_face["bbox"])
        
        arc_face_embeddings = (
            arc_face_feats
            + arc_face_e_i
            + arc_face_e_t
            + arc_face_type_embeddings
            + arc_face_cluster_embeddings 
            + arc_face_bbox_embeddings
        )


        video_masks = torch.zeros(
            (self.batch_size, self.nvid * 5), dtype=torch.float32
        ).to(device="cuda")
        # [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        video_segments = torch.tensor([i for i in range(5)] * 5).repeat(
            self.batch_size, 1
        )
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
        video_ids = torch.tensor([i for i in range(5) for j in range(5)]).repeat(
            self.batch_size, 1
        )
        video_e_t = self.segment_embed(video_segments.to(device="cuda"))
        video_e_i = video_embed(video_ids.to(device="cuda"))
        video_type = torch.full_like(video_ids, 1).to(device="cuda")
        video_type_embeddings = self.type_embed(video_type)
        video_embeddings = (
            fc["embedding"] + video_e_i + video_e_t + video_type_embeddings
        )
       
        for i, sent_num in enumerate(slot_sent["sent_num"]):
            video_masks[i, : sent_num * 5] = 1

        
        if text_embeddings is not None:
            encoder_input = torch.cat(
                (
                    text_embeddings,
                    video_embeddings,
                    arc_face_embeddings,
                    video_clip_embeddings,
                ),
                dim=1,
            )
            input_masks = torch.cat(
                (text_masks, video_masks, arc_face["masks"], video_clip["masks"]), dim=1
            )

        input_masks = ~input_masks.bool()
        encoder_output = self.transformer_encoder(encoder_input.transpose(0,1), src_key_padding_mask = input_masks)


        return encoder_output, input_masks


    def forward_caption(
        self,
        arc_face,
        video_clip,
        fc,
        slot,
        slot_sent,
        video_embed,
    ):
        
        self.batch_size = arc_face["feats"].shape[0]
        


        video_clip["segments"] = video_clip["segments"].reshape(
            self.batch_size, video_clip["segments"].shape[1] * video_clip["segments"].shape[2]
        )
        video_clip_e_t = self.segment_embed(video_clip["segments"])
        video_clip_ids = torch.tensor([i for i in range(5) for j in range(50)]).repeat(
            self.batch_size, 1
        )
        video_clip_e_i = video_embed(video_clip_ids.to(device="cuda"))
        if self.config.run_type == "fc":
            video_clip_type = torch.full_like(video_clip_ids, 2).to(device="cuda")
        else:
            video_clip_type = torch.full_like(video_clip_ids, 3).to(device="cuda")
        video_clip_type_embeddings = self.type_embed(video_clip_type)

        video_clip_feats = self.clip_encode(video_clip["feats"])

        video_clip_embeddings = (
            video_clip_feats
            + video_clip_e_i
            + video_clip_e_t
            + video_clip_type_embeddings
        )
        video_clip["masks"] = video_clip["masks"].reshape(
            self.batch_size, video_clip["masks"].shape[1] * video_clip["masks"].shape[2]
        )

        arc_face["segments"] = arc_face["segments"].to(torch.long)
        arc_face["vid_ids"] = arc_face["vid_ids"].to(torch.long)
        arc_face["clusters"] = arc_face["clusters"].to(torch.long)
        arc_face_e_t = self.segment_embed(arc_face["segments"])
        arc_face_e_i = video_embed(arc_face["vid_ids"])
        if self.config.run_type == "fc":
            arc_face_type = torch.full_like(arc_face["vid_ids"], 1).to(device="cuda")
        else:
            arc_face_type = torch.full_like(arc_face["vid_ids"], 2).to(device="cuda")
        arc_face_type_embeddings = self.type_embed(arc_face_type)

        arc_face_feats = self.arc_face_encoding(arc_face["feats"])

        arc_face_cluster_embeddings = self.arc_cluster_embed(arc_face["clusters"])
        arc_face_bbox_embeddings = self.arc_bbox_encode(arc_face["bbox"])
        
        arc_face_embeddings = (
            arc_face_feats
            + arc_face_e_i
            + arc_face_e_t
            + arc_face_type_embeddings
            + arc_face_cluster_embeddings 
            + arc_face_bbox_embeddings
        )

        
        video_masks = torch.zeros(
            (self.batch_size, self.nvid * 5), dtype=torch.float32
        ).to(device="cuda")
        # [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        video_segments = torch.tensor([i for i in range(5)] * 5).repeat(
            self.batch_size, 1
        )
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
        video_ids = torch.tensor([i for i in range(5) for j in range(5)]).repeat(
            self.batch_size, 1
        )
        video_e_t = self.segment_embed(video_segments.to(device="cuda"))
        video_e_i = video_embed(video_ids.to(device="cuda"))
        if self.config.run_type == "fc":
            video_type = torch.full_like(video_ids, 0).to(device="cuda")
        else:
            video_type = torch.full_like(video_ids, 1).to(device="cuda")
        video_type_embeddings = self.type_embed(video_type)
        video_embeddings = (
            fc["embedding"] + video_e_i + video_e_t + video_type_embeddings
        )

        for i, sent_num in enumerate(slot_sent["sent_num"]):
            video_masks[i, : sent_num * 5] = 1

    
        encoder_input = torch.cat(
            (
                video_embeddings,
                arc_face_embeddings,
                video_clip_embeddings
            ),
            dim=1,
        )
        input_masks = torch.cat(
            (video_masks, arc_face["masks"], video_clip["masks"]), dim=1
        )

        input_masks = ~input_masks.bool()

        encoder_output = self.transformer_encoder(encoder_input.transpose(0,1), src_key_padding_mask = input_masks)

        return encoder_output, input_masks

        
        