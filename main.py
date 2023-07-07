
"""Two things to finetune """
### 1. The noise generator
### 2. The text encoder
### 3. Diffusion model

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from utils.word_vectorizer import WordVectorizer, POS_enumerator
 
# from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from models import *

PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77
DIM_WORD = 300
DIM_POS_OHOT = len(POS_enumerator)


if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        num_classes = 200 // opt.unit_length
        meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'Comp_v6_KLD01', 'meta')
elif opt.dataset_name == 'kit':
    opt.data_root = './dataset/KIT-ML'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.joints_num = 21
    radius = 240 * 8
    fps = 12.5
    dim_pose = 251
    opt.max_motion_length = 196
    num_classes = 200 // opt.unit_length
    meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'Comp_v6_KLD005', 'meta')

else:
    pass

class SleepWalkerTrainer(tf.keras.Model):
    # Reference:
    # https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

    def __init__(
        self,
        motion_encoder, #our motion encoder model
        movement_encoder,# our movement enocder
        diffusion_model,#
        noise_scheduler,
        opt,
        use_mixed_precision=False,
        prior_loss_weight=1.0,
        max_grad_norm=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.movement_encoder = movement_encoder
        self.motion_encoder = motion_encoder
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
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

    def train_step(self, inputs):  # sourcery skip: extract-method
        # instance_batch = inputs[0]
        # class_batch = inputs[1]
        # instance_images = instance_batch["instance_images"]
        # instance_texts = instance_batch["instance_texts"]
        # class_images = class_batch["class_images"]
        # class_texts = class_batch["class_texts"]
        instance_word_emb,  instance_motions = inputs[0] #batch_data
        class_word_emb, class_pos_ohot, class_caption, class_cap_lens, class_motions, class_m_lens, _ = inputs[1] #batch_data
        #word_emb, #pos_ohot, #caption, #cap_lens, #motions, #m_lens, _ = batch_data
        

        motions = tf.concat([instance_motions, class_motions], 0)
        texts = tf.concat(
            [instance_word_emb, class_word_emb], 0
        )  # `texts` can either be caption tokens or embedded caption tokens.
        batch_size = tf.shape(motions)[0]

        with tf.GradientTape() as tape:
            # If the `text_encoder` is being fine-tuned.
            if self.train_text_encoder:
                texts = self.text_encoder(texts, self.pos_ohot, self.cap_lens)

            # Project image into the latent space and sample from it.
            movements_latents  = self.movement_enc(motions[..., :-4])
            motion_latents = self.motion_encoder(movements_latents, self.m_lens)

            latents = self.sample_from_encoder_outputs(motion_latents)
            # Know more about the magic number here:
            # https://keras.io/examples/generative/fine_tune_via_textual_inversion/
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents.
            noise = tf.random.normal(tf.shape(latents))

            # Sample a random timestep for each image.
            timesteps = tnp.random.randint(
                0, self.noise_scheduler.train_timesteps, (batch_size,)
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process).
            noisy_latents = self.noise_scheduler.add_noise(
                tf.cast(latents, noise.dtype), noise, timesteps
            )
            # Get the target for loss depending on the prediction type
            # just the sampled noise for now.
            target = noise  # noise_schedule.predict_epsilon == True

            # Predict the noise residual and compute loss.
            timestep_embedding = tf.map_fn(
                lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
            )
            
            model_pred = self.diffusion_model.generate(
                [noisy_latents, timestep_embedding, texts], training=True
            )
            loss = self.compute_loss(target, model_pred)
            if self.use_mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # Update parameters of the diffusion model.
        if self.train_text_encoder:
            trainable_vars = (
                self.text_encoder.trainable_variables
                + self.diffusion_model.trainable_variables
            )
        else:
            trainable_vars = self.diffusion_model.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)
        if self.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {m.name: m.result() for m in self.metrics}

    def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
        half = dim // 2
        log_max_preiod = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(-log_max_preiod * tf.range(0, half, dtype=tf.float32) / half)
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        return embedding

    def sample_from_encoder_outputs(self, outputs):
        mean, logvar = tf.split(outputs, 2, axis=-1)
        logvar = tf.clip_by_value(logvar, -30.0, 20.0)
        std = tf.exp(0.5 * logvar)
        sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + std * sample

    def compute_loss(self, target, model_pred):
        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
        model_pred, model_pred_prior = tf.split(model_pred, num_or_size_splits=2, axis=0)
        target, target_prior = tf.split(target, num_or_size_splits=2, axis=0)

        # Compute instance loss.
        loss = self.compiled_loss(target, model_pred)

        # Compute prior loss.
        prior_loss = self.compiled_loss(target_prior, model_pred_prior)

        # Add the prior loss to the instance loss.
        loss = loss + self.prior_loss_weight * prior_loss
        return loss

    def save_weights(
        self, ckpt_path_prefix, overwrite=True, save_format=None, options=None
    ):
        # Overriding this method will allow us to use the `ModelCheckpoint`
        # callback directly with this trainer class. In this case, it will
        # only checkpoint the `diffusion_model` and optionally the `text_encoder`.
        diffusion_model_path = ckpt_path_prefix + "-unet.h5"
        self.diffusion_model.save_weights(
            filepath=diffusion_model_path,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.diffusion_model_path = diffusion_model_path
        if self.train_text_encoder:
            text_encoder_model_path = ckpt_path_prefix + "-text_encoder.h5"
            self.text_encoder.save_weights(
                filepath=text_encoder_model_path,
                overwrite=overwrite,
                save_format=save_format,
                options=options,
            )
            self.text_encoder_model_path = text_encoder_model_path
