from networks.modules import *
import torch
from utils.word_vectorizer import WordVectorizer, POS_enumerator


def build_models(opt):
        if opt.text_enc_mod == 'bigru':
            text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                            pos_size=opt.dim_pos_ohot,
                                            hidden_size=opt.dim_text_hidden,
                                            device=torch.device("cpu"))

            text_size = opt.dim_text_hidden * 2
        else:
            raise Exception("Text Encoder Mode not Recognized!!!")

        seq_prior = TextDecoder(text_size=opt.text_size,
                                input_size=opt.dim_att_vec + opt.dim_movement_latent,
                                output_size=opt.dim_z,
                                hidden_size=opt.dim_pri_hidden,
                                n_layers=opt.n_layers_pri)


        seq_decoder = TextVAEDecoder(text_size=opt.text_size,
                                    input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                    output_size=opt.dim_movement_latent,
                                    hidden_size=opt.dim_dec_hidden,
                                    n_layers=opt.n_layers_dec)

        att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                            key_dim=opt.text_size,
                            value_dim=opt.dim_att_vec)

        movement_encoder = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
        movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

        # latent_dis = LatentDis(input_size=opt.dim_z * 2)

        # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
        return text_encoder, seq_prior, seq_decoder, att_layer, movement_encoder, movement_dec
