import torch
import torch.nn as nn
from .ftv_encoder import FTV_encoder
from transformers import BertTokenizer
import os



class MSTModel(nn.Module):
    def __init__(self, opt):
        super(MSTModel, self).__init__()
        self.memory_encoding_size = opt.encoding_size
        print(f"Mem encoding size is : {self.memory_encoding_size}")
        self.batch_size = opt.batch_size
        self.max_caption_length = 120
        self.max_cluster_possible = 50

        self.opt = opt

        self.use_text_encoder = True

        # Bert Modelling
        if self.use_text_encoder:
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.opt.data_dir, self.opt.tokenizer_path))
            self.TOKEN_LENGTH = len(
                self.tokenizer
            ) 
            print(f"Vocab size is : {self.TOKEN_LENGTH}")
            self.CLS_TOKEN_ID = 101
            self.SEP_TOKEN_ID = 102
            self.START_TOKEN = 30536
            self.END_TOKEN_cap = 30537
            self.END_TOKEN = 13
            self.MAX_CAPTIONS = 5

        # Caption Embedding
        self.caption_embedding = nn.Embedding(
            self.TOKEN_LENGTH, self.memory_encoding_size
        ) 

        # BERT Encode
        self.bert_encode = nn.Linear(opt.bert_size, self.memory_encoding_size)

        # Caption Position + Segment Embeddings
        self.position_embed = nn.Embedding(
            self.max_caption_length, self.memory_encoding_size
        )
        self.segment_embed = nn.Embedding(6, self.memory_encoding_size)

        # Feature Conversion
        self.video_encode = nn.Linear(1024, self.memory_encoding_size)


        # Character Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.memory_encoding_size, nhead=8
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=3
        )

        num_classes = opt.unique_characters + 1

        # Encoder
        self.encoder = FTV_encoder(opt)

        if opt.run_type == "fitb":
            self.dense = nn.Linear(self.memory_encoding_size, num_classes)
            self.seen_dense = nn.Linear(self.memory_encoding_size, 2)
            
        if opt.run_type == "fc" or opt.run_type == "joint":
            self.vocab_dense = nn.Linear(self.memory_encoding_size,self.TOKEN_LENGTH)

        self.logits = nn.Linear(self.memory_encoding_size, self.TOKEN_LENGTH)

        # Loss function
        self.softmax = nn.Softmax(dim=1)
        self.cross_loss = nn.CrossEntropyLoss()
    
    def full_caption_autoregressive_decoder(
        self,
        encoder_output,
        gt_captions,
        encoder_masks,
    ):
        predictions = torch.zeros(
            (self.batch_size, self.max_caption_length), dtype=int
        ).cuda()


        pos_tensor = torch.arange(0, 120)
        pos_ids = pos_tensor.expand(self.batch_size,-1).cuda()



        predictions[:, 0] = gt_captions[:, 0]

        batch_size = predictions.shape[0]

        end_token_generated = torch.zeros(batch_size, dtype=torch.bool).cuda()
        sep_token_count = torch.zeros(batch_size, dtype=torch.long).cuda()
        sep_token_embedding_input = torch.zeros(batch_size, self.max_caption_length, dtype=torch.long).cuda()


        index = 1

        while index < self.max_caption_length and not torch.all(end_token_generated):
            
            predictions_emb = self.caption_embedding(
                predictions[:, : index]
            )  
            seg_ids_subset = sep_token_embedding_input[:,: index]
            seg_embeddings = self.segment_embed(seg_ids_subset)

            pos_ids_subset = pos_ids[:,:index]
            pos_embeddings = self.position_embed(pos_ids_subset)

            predictions_emb = (
                predictions_emb
                + pos_embeddings
                + seg_embeddings
            )


            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                predictions_emb.shape[1]
            ).cuda()
            dec_output = self.transformer_decoder(
                predictions_emb.transpose(0, 1),
                encoder_output.transpose(0, 1),
                tgt_mask=tgt_mask,
                memory_key_padding_mask=encoder_masks,
            ).transpose(0, 1)

            final_output = dec_output[:, -1, :]

            
            logits_vocab = self.vocab_dense(final_output)

            _, tgt_vocab = torch.max(logits_vocab, dim=1)

            next_tokens = torch.zeros(predictions.shape[0], dtype=torch.long).cuda()

            next_tokens = tgt_vocab

            predictions[~end_token_generated, index] = next_tokens[~end_token_generated]

            generated_sep = (predictions[:, index] == self.SEP_TOKEN_ID)
            sep_token_count += generated_sep.long()

            sep_token_embedding_input[:, index] = sep_token_count

            # Add end token for those that have just reached 5 SPECIAL_TOKEN occurrences but haven't generated end token yet
            reached_five_sep = (sep_token_count == 5) & (~end_token_generated)

            if reached_five_sep.any() and index + 1 < self.max_caption_length:
                predictions[reached_five_sep, index + 1] = 30537
                end_token_generated |= reached_five_sep

            is_end_token = (predictions[:, index] == self.END_TOKEN_cap)

            predictions[is_end_token, index] = 30537

            end_token_generated |= is_end_token

            index += 1
    
        predictions[:,0] = 30536
        return predictions


    def fill_in_autoregressive_decoder(
        self,
        encoder_output,
        gt_captions,
        caption_pos_ids,
        caption_seg_ids,
        encoder_masks,
        blank_indexes,
    ):
        predictions = torch.zeros(
            (self.batch_size, self.max_caption_length), dtype=int
        ).cuda()
        predictions[:, 0] = gt_captions[:, 0]
        

        for i, gt_caption in enumerate(gt_captions):
            single_encoder_output = encoder_output[i]  # 92 x 512
            encoder_mask = encoder_masks[i]
            blank_index = (blank_indexes[i] - 1).tolist()
            for index, token in enumerate(gt_caption):
            
                if token.item() == self.END_TOKEN or token.item() == self.END_TOKEN_cap:
                    break
    
                if index in blank_index:
                    
                    predictions_emb = self.caption_embedding(
                        predictions[i, : index + 1]
                    )  

                    predictions_emb = (
                        predictions_emb
                        + caption_pos_ids[i, : index + 1]
                        + caption_seg_ids[i, : index + 1]
                    )
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        predictions_emb.shape[0]
                    ).cuda()
                    dec_output = self.transformer_decoder(
                        predictions_emb,
                        single_encoder_output,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=encoder_mask,
                    )
                    if self.opt.run_type == "fitb":
                        logits = self.dense(dec_output)
                    else: 
                        logits = self.vocab_dense(dec_output)
                    _, tgt = torch.max(logits, 1)
                    next_token = tgt[-1]
                    predictions[i, index + 1] = next_token 
                else:
                    tgt = gt_caption[index + 1]
                    predictions[i, index + 1] = tgt
            

        return predictions
    


    def forward(self, *args, **kwargs):
        mode = kwargs.get("mode", "forward")
        if "mode" in kwargs:
            del kwargs["mode"]
        return getattr(self, "_" + mode)(*args, **kwargs)

   
    def _forward_fitb(
        self,
        nba,
        arc_face,
        video_clip,
        fc,
        slot,
        slot_sent,
    ):
        
        self.batch_size = arc_face["feats"].shape[0]
        
        nba["blank_masks"][:, -1].fill_(0)

        

        gt_mask = nba["blank_masks"].bool()
        
        column_of_zeros = torch.zeros(nba["blank_masks"].shape[0], 1).cuda()
        pred_mask = torch.cat((nba["blank_masks"][:, 1:], column_of_zeros), dim=1)

       

        video_clip["feats"] = video_clip["feats"].reshape(
            self.batch_size,
            video_clip["feats"].shape[1] * video_clip["feats"].shape[2],
            video_clip["feats"].shape[3],
        )
        

        fc["embedding"] = self.video_encode(fc["feats"])
        fc["embedding"] = fc["embedding"].reshape(
            self.batch_size,
            fc["embedding"].shape[1] * fc["embedding"].shape[2],
            fc["embedding"].shape[3],
        )

        text_embedding = self.bert_encode(nba["bert_emb"])  


        encoder_output, input_masks = self.encoder.forward(
            dict(arc_face),
            dict(video_clip),
            dict(fc),
            text_embedding,
            dict(slot),
            dict(slot_sent),
            self.segment_embed,
        )
        
        if self.opt.run_type == "fitb":
            nba["gt_captions"] = torch.where(
                nba["gt_captions"] >= 30524, nba["gt_captions"] - 30524, nba["gt_captions"]
            )

        # bs x 120
        gt_caption_embedding = self.caption_embedding(nba["gt_captions"])
        caption_pos_ids = self.position_embed(nba["position_ids"])
        
        caption_seg_ids = self.segment_embed(nba["segment_ids"])

        gt_caption_embedding = gt_caption_embedding + caption_pos_ids + caption_seg_ids
        
        tgt_key_padding_mask = ~nba["gt_masks"].bool().cuda()
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            gt_caption_embedding.shape[1]
        ).cuda()
        input_masks = input_masks.cuda()
        
        dec_output = self.transformer_decoder(
            gt_caption_embedding.transpose(0, 1),
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=input_masks,
        ).transpose(0, 1)
       

        if self.opt.run_type == "fitb":
            logits = self.dense(dec_output[pred_mask.bool()])
        else: 
            logits = self.vocab_dense(dec_output[pred_mask.bool()])
       

        labels = nba["gt_captions"].clone()
        labels = labels[gt_mask]

        loss = self.cross_loss(logits, labels)

        return loss


    def _predict_fitb(
            self,
            nba,
            arc_face,
            video_clip,
            fc,
            slot,
            slot_sent
        ):
        
        self.batch_size = arc_face["feats"].shape[0]
       

        
        text_embedding = self.bert_encode(nba["bert_emb"])  # 17 x 1536 --> 17 x 512

        video_clip["feats"] = video_clip["feats"].reshape(
            self.batch_size,
            video_clip["feats"].shape[1] * video_clip["feats"].shape[2],
            video_clip["feats"].shape[3],
        )
       
        # gt_caption_embedding = self.text_encoder_embedding_layer(captions)
        if self.opt.run_type == "fitb":
            nba["gt_captions"] = torch.where(
                nba["gt_captions"] >= 30524, nba["gt_captions"] - 30524, nba["gt_captions"]
            )


        # bs x 120
        gt_caption_embedding = self.caption_embedding(nba["gt_captions"])
        caption_pos_ids = self.position_embed(nba["position_ids"])
        caption_seg_ids = self.segment_embed(nba["segment_ids"])

        gt_caption_embedding = gt_caption_embedding + caption_pos_ids + caption_seg_ids

        
        
        fc["embedding"] = self.video_encode(fc["feats"])
        fc["embedding"] = fc["embedding"].reshape(
            self.batch_size,
            fc["embedding"].shape[1] * fc["embedding"].shape[2],
            fc["embedding"].shape[3],
        )
        
        encoder_output, input_masks = self.encoder.forward(
            dict(arc_face),
            dict(video_clip),
            dict(fc),
            text_embedding,
            dict(slot),
            dict(slot_sent),
            self.segment_embed,
        )

        input_masks = input_masks.cuda()

        
        predicted_caption = self.fill_in_autoregressive_decoder(
            encoder_output.transpose(0, 1),
            nba["gt_captions"],
            caption_pos_ids,
            caption_seg_ids,
            input_masks,
            nba["blank_indexes"],
        )
        predicted_person_tokens = torch.masked_select(
            predicted_caption, nba["blank_masks"].bool()
        )
        predictions = predicted_person_tokens
        print(f"Predictions : {predictions}")
 
        predicted_genders = predictions.new_zeros(
            predictions.size(0), dtype=torch.long
        )

        return predictions, predicted_genders
    

    def _forward_caption(
        self,
        nba,
        arc_face,
        video_clip,
        fc,
        slot,
        slot_sent
    ):
        
        self.batch_size = arc_face["feats"].shape[0]
        
        nba["blank_masks"][:, -1].fill_(0)
        

        gt_mask = nba["blank_masks"].bool()
        # Pred_mask is shifted to left by one position.
        column_of_zeros = torch.zeros(nba["blank_masks"].shape[0], 1).cuda()
        pred_mask = torch.cat((nba["blank_masks"][:, 1:], column_of_zeros), dim=1)


        video_clip["feats"] = video_clip["feats"].reshape(
            self.batch_size,
            video_clip["feats"].shape[1] * video_clip["feats"].shape[2],
            video_clip["feats"].shape[3],
        )

        fc["embedding"] = self.video_encode(fc["feats"])
        fc["embedding"] = fc["embedding"].reshape(
            self.batch_size,
            fc["embedding"].shape[1] * fc["embedding"].shape[2],
            fc["embedding"].shape[3],
        )

        # bs x 92 x 512 [ 25 + 50 + 17]
        encoder_output, input_masks = self.encoder.forward_caption(
            dict(arc_face),
            dict(video_clip),
            dict(fc),
            dict(slot),
            dict(slot_sent),
            self.segment_embed,
        )
        
        gt_caption_embedding = self.caption_embedding(nba["gt_captions"])
        caption_pos_ids = self.position_embed(nba["position_ids"])
        caption_seg_ids = self.segment_embed(nba["segment_ids"])

        gt_caption_embedding = gt_caption_embedding + caption_pos_ids + caption_seg_ids
        

      
        tgt_key_padding_mask = ~nba["gt_masks"].bool().cuda()
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            gt_caption_embedding.shape[1]
        ).cuda()
        input_masks = input_masks.cuda()


        dec_output = self.transformer_decoder(
            gt_caption_embedding.transpose(0, 1),
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=input_masks,
        ).transpose(0, 1)
        

        # Masks for full captioning

        label_mask = nba["gt_masks"].clone()
        label_mask[:,0] = 0

        final_label_mask = label_mask.bool() * ~gt_mask.bool()

        output_mask =   nba["gt_masks"].clone()
        column_of_zeros = torch.zeros(output_mask.shape[0], 1).cuda()
        output_mask = torch.cat((output_mask[:, 1:], column_of_zeros), dim=1)

        final_output_mask = output_mask.bool() * ~pred_mask.bool()

        
        logits = self.vocab_dense(dec_output[pred_mask.bool()])
        
       
        labels = nba["gt_captions"].clone()
        labels = labels[gt_mask]


        loss = self.cross_loss(logits, labels)

        
        fc_logits = self.vocab_dense(dec_output[final_output_mask.bool()])
        

        fc_labels = nba["gt_captions"].clone()
        fc_labels = fc_labels[final_label_mask.bool()]

        fc_loss = self.cross_loss(fc_logits,fc_labels)

        
        return loss + fc_loss
    
    def _predict_caption(
        self,
        nba,
        arc_face,
        video_clip,
        fc,
        slot,
        slot_sent
    ):


        self.batch_size = arc_face["feats"].shape[0]

        video_clip["feats"] = video_clip["feats"].reshape(
            self.batch_size,
            video_clip["feats"].shape[1] * video_clip["feats"].shape[2],
            video_clip["feats"].shape[3],
        )
        
        gt_caption_embedding = self.caption_embedding(nba["gt_captions"])
        caption_pos_ids = self.position_embed(nba["position_ids"])
        caption_seg_ids = self.segment_embed(nba["segment_ids"])

        gt_caption_embedding = gt_caption_embedding + caption_pos_ids + caption_seg_ids

    
        fc["embedding"] = self.video_encode(fc["feats"])
        fc["embedding"] = fc["embedding"].reshape(
            self.batch_size,
            fc["embedding"].shape[1] * fc["embedding"].shape[2],
            fc["embedding"].shape[3],
        )

        
        encoder_output, input_masks = self.encoder.forward_caption(
            dict(arc_face),
            dict(video_clip),
            dict(fc),
            dict(slot),
            dict(slot_sent),
            self.segment_embed,
        )

        input_masks = input_masks.cuda()

        predicted_caption = self.full_caption_autoregressive_decoder(
            encoder_output.transpose(0, 1),
            nba["gt_captions"],
            input_masks,
        )

        return predicted_caption

        
