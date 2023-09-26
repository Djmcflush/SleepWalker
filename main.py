
"""Two things to finetune """
### 1. The noise generator
### 2. The text encoder
### 3. Diffusion model
from data.dataset import RawTextDataset, Text2MotionDataset
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.opt import OptModel
from utils.build import build_models
# from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from models import *
from os.path import join as pjoin


PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77
DIM_WORD = 300
DIM_POS_OHOT = len(POS_enumerator)

DIM_POSE = 263
META_ROOT = []
NUM_CLASSES = 0

class SleepWalkerTrainer(tf.keras.Model):
    # Reference:
    # https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
    #the text to motion generation model is: ./checkpoints/t2m/Comp_v6_KLD01/           # Text-to-motion generation model

    
    def __init__(
        self,
        movement_encoder,# our movement enocder
        diffusion_model,#CompTrainerV6
        opt,
        use_mixed_precision=False,
        prior_loss_weight=1.0,
        max_grad_norm=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        # if opt.dataset_name == 't2m':

        #NUM_CLASSES = 200 // opt.unit_length
        #META_ROOT = pjoin(opt.checkpoints_dir, opt.dataset_name, 'Comp_v6_KLD01', 'meta')

        self.movement_encoder = movement_encoder
        self.diffusion_model = diffusion_model
        #Only finetune text_encoder if user wishes to do so
        if self.train_text_encoder:
            self.text_encoder = TextEncoderBiGRUCo(word_size=DIM_WORD,
                                  pos_size=DIM_POS_OHOT,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)
            self.text_encoder.trainable = True
            self.pos_ids = tf.convert_to_tensor(
                [list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32
            )

        self.prior_loss_weight = prior_loss_weight
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision

    
    def generate_inversion(self, word_hids, hidden, motions, m_lens):

        '''
        Inversion Generation (generate_inversion method): This method generates 
        the inversion of the model. It takes word hidden states, hidden states, motions,
        and motion lengths as inputs. It encodes the movements, generates the local attention vector, 
        computes the posterior and prior, and decodes the movements. It also keeps track of the
        generated movements and attention weights.
        '''
        # Initially input a mean vector
        movements_latents  = self.movement_enc(motions[..., :-4])
        mov_len = movements_latents.shape[1]
        mov_in = self.sample_from_encoder_outputs(movements_latents)
        latents = latents * 0.18215
        target = tf.random.normal(tf.shape(latents))

        hidden_pri = self.seq_pri.get_init_hidden(hidden)
        hidden_dec = self.seq_dec.get_init_hidden(hidden)

        mus_pri = []
        logvars_pri = []
        fake_mov_batch = []
        att_wgt = []
        mus_post = []
        logvars_post = []

        query_input = []
        for i in range(mov_len):

            mov_tgt = self.movements[:, i]

            '''Local Attention Vector'''
            att_vec, co_weights = self.diffusion_model.att_layer(hidden_dec[-1], word_hids)
            query_input.append(hidden_dec[-1])

            tta = m_lens // self.opt.unit_length - i
            # tta = m_lens - i


            if self.opt.text_enc_mod == 'bigru':
                pos_in = torch.cat([mov_in, mov_tgt, att_vec], dim=-1)
                pri_in = torch.cat([mov_in, att_vec], dim=-1)

            elif self.opt.text_enc_mod == 'transformer':
                pos_in = torch.cat([mov_in, mov_tgt, att_vec.detach()], dim=-1)
                pri_in = torch.cat([mov_in, att_vec.detach()], dim=-1)

            '''Posterior'''
            z_pos, mu_pos, logvar_pos, hidden_pos = self.diffusion_model.seq_post(pos_in, hidden_pos, tta)

            '''Prior'''
            z_pri, mu_pri, logvar_pri, hidden_pri = self.diffusion_model.seq_pri(pri_in, hidden_pri, tta)

            '''Decoder'''
            dec_in = torch.cat([mov_in, att_vec, z_pos], dim=-1) #Train mode
            # dec_in = torch.cat([mov_in, att_vec, z_pri], dim=-1) #Eval Mode
            fake_mov, hidden_dec = self.diffusion_model.seq_dec(dec_in, mov_in, hidden_dec, tta)

            # print(fake_mov.shape)

            mus_post.append(mu_pos)
            logvars_post.append(logvar_pos)
            mus_pri.append(mu_pri)
            logvars_pri.append(logvar_pri)
            fake_mov_batch.append(fake_mov.unsqueeze(1))
            att_wgt.append(co_weights)

            mov_in = fake_mov.detach()

        self.diffusion_model.fake_movements = torch.cat(fake_mov_batch, dim=1)
        self.diffusion_model.att_wgts = torch.cat(att_wgt, dim=-1)

        # print(self.fake_movements.shape)

        self.diffusion_model.fake_motions = self.diffusion_model.mov_dec(self.diffusion_model.fake_movements)

        self.diffusion_model.mus_pri = torch.cat(mus_pri, dim=0)
        self.diffusion_model.logvars_pri = torch.cat(logvars_pri, dim=0)
        self.diffusion_model.mus_post = torch.cat(mus_post, dim=0)
        self.diffusion_model.mus_pri = torch.cat(mus_pri, dim=0)
        self.diffusion_model.logvars_post = torch.cat(logvars_post, dim=0)
        self.diffusion_model.logvars_pri = torch.cat(logvars_pri, dim=0)

        return self.diffusion_model.fake_motions, self.diffusion_model.fake_movements, target    


    def sample_from_encoder_outputs(self, outputs):
        """"Sampling from Encoder Outputs (sample_from_encoder_outputs method): 
        This method takes the outputs of an encoder, splits them into mean and logvar, 
        generates a random sample, and returns a sampled output by combining the mean and standard deviation."""
        
        # Flatten the tensor except for the last dimension
        last_dim = tf.shape(outputs)[-1]
        flattened_shape = tf.concat([[-1], [last_dim]], axis=0)
        reshaped_outputs = tf.reshape(outputs, flattened_shape)

        # Split the flattened tensor into mean and logvar
        mean, logvar = tf.split(reshaped_outputs, 2, axis=-1)

        # Clip the logvar values to a specific range
        logvar = tf.clip_by_value(logvar, -30.0, 20.0)

        # Calculate the standard deviation from logvar
        std = tf.exp(0.5 * logvar)

        # Generate a random sample with the same shape as the mean tensor
        sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)

        # Reshape the sample back to the original N-dimensional shape
        original_shape = tf.shape(outputs)
        reshaped_sample = tf.reshape(sample, original_shape)

        # Calculate the final sampled output by combining mean and standard deviation
        return mean + std * reshaped_sample

    def train_step(self, inputs):  # sourcery skip: extract-method
        '''Define TrainStep for Sleepwalker. Here variables with the term instance mean motions introduced by the user. 
        Variables with the term class as a prefix are generated by the diffusion model for class priors'''
        #Algorithim
        # Encode motions
        # sample latent from motion latents
        # create noise from motion output latents
        # decode text encodings to motioons
        # compute loss between 

        instance_batch = inputs[0]
        class_batch = inputs[1]

        instance_word_emb, instance_pos_ohot, instance_caption, instance_cap_lens, instance_motions, instance_m_lens = instance_batch #3-10
        class_word_emb, class_pos_ohot, class_caption, class_cap_lens, class_motions, class_m_lens = class_batch #200-300
        

        motions = tf.concat([instance_motions, class_motions], 0)
        texts = tf.concat(
            [instance_word_emb, class_word_emb], 0
        )  # `texts` can either be caption tokens or embedded caption tokens.
        pos_ohot = tf.concat([instance_pos_ohot, class_pos_ohot], 0)
        # captions = tf.concat([instance_caption, class_caption], 0)
        cap_lens = tf.concat([instance_cap_lens, class_m_lens],0)
        m_lens = instance_m_lens + class_m_lens
        # batch_size = tf.shape(motions)[0]
        word_hids, hidden = self.text_enc(texts, pos_ohot, cap_lens)

        with tf.GradientTape() as tape:
            
            fake_motions, fake_movements, target_movements  =  self.generate_inversion(word_hids, hidden, motions, m_lens)
            log_dict = self.diffusion_model.update() # does loss of mov and loss of motions Then updates gradients

            ## Compute loss between model prediction and target
            # loss_mov = self.compute_loss(target_movements, fake_movements)
            # loss_mot = self.compute_loss(motions, fake_motions)
           
        return log_dict

    def save_weights(
        self, filename, epoch, total_it, sub_ep, sl_len
    ):
        ##Save model State
        self.diffusion_model.save(filename, epoch, total_it, sub_ep, sl_len)
        
    
if __name__ == "__main__":
    # Initialize the SleepWalkerTrainer
    opt =OptModel()
    
    text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec = build_models(opt)
    trainer = diffusion(text_enc=text_enc,seq_pri=seq_pri,seq_dec=seq_dec,att_layer=att_layer,mov_enc=mov_enc,mov_dec=mov_dec, args=opt)
    diffusion_model = trainer.load(pjoin(opt.model_dir, 'latest.tar'))
    sleep_walker_trainer = SleepWalkerTrainer(movement_encoder=mov_enc,
                                              diffusion_model=diffusion_model,
                                              opt=opt
                                              )
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    dataset_new = Text2MotionDataset(opt, np.load(opt.mean), np.load(opt.st), opt.text_file, w_vectorizer)
    dataset_old = Text2MotionDataset(opt, np.load(opt.mean), np.load(opt.st), opt.text_file, w_vectorizer)
    new_motions_dataset = DataLoader(dataset_new, batch_size=1, drop_last=True, num_workers=1)
    old_motions_dataset = DataLoader(dataset_old, batch_size=1, drop_last=True, num_workers=1)
    
    inputs = (new_motions_dataset, old_motions_dataset)
    # Train the model
    sleep_walker_trainer.train_step(inputs)


