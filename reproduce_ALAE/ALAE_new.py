from models import *
from custom_adam import LREQAdam
from tqdm import tqdm
from torchvision.utils import save_image
import os
from tracker import LossTracker

# import torch
# torch.manual_seed(0)

def compute_r1_gradient_penalty(d_result_real, real_images):
    # real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
    real_grads = torch.autograd.grad(d_result_real, real_images,
                                     grad_outputs=torch.ones_like(d_result_real),
                                     create_graph=True, retain_graph=True)[0]
    r1_penalty = 0.5 * torch.sum(real_grads.pow(2.0), dim=[1, 2, 3]) # Norm on all dims but batch

    return r1_penalty


class ALAE:
    def __init__(self, latent_size, device, hyper_parameters):
        self.device = device
        self.latent_size = latent_size
        self.hp = {'lr': 0.002, "batch_size": 128, 'mapping_layers': 6, "g_penalty_coeff": 10}
        self.hp.update(hyper_parameters
                       )
        self.D = DiscriminatorFC(w_dim=latent_size,mapping_layers=3).to(device).train()

        self.F = VAEMappingFromLatent(z_dim=latent_size, w_dim=latent_size, mapping_layers=self.hp['mapping_layers']).to(device).train()

        self.G = GeneratorFC(latent_size=latent_size).to(device).train()

        self.E = EncoderFC(latent_size=latent_size).to(device).train()

        self.ED_optimizer = LREQAdam(list(self.E.parameters()) + list(self.D.parameters()), lr=self.hp['lr'], betas=(0.0, 0.99), weight_decay=0)
        self.FG_optimizer = LREQAdam(list(self.F.parameters()) + list(self.G.parameters()), lr=self.hp['lr'], betas=(0.0, 0.99), weight_decay=0)
        self.EG_optimizer = LREQAdam(list(self.E.parameters()) + list(self.G.parameters()), lr=self.hp['lr'], betas=(0.0, 0.99), weight_decay=0)

    def generate(self, z):
        self.F.eval()
        self.G.eval()

        w_vector = self.F(z)
        image = self.G(w_vector)

        self.F.train()
        self.G.train()

        return image

    def encode(self, imgs):
        self.E.eval()
        w_vector = self.E(imgs)
        self.E.train()

        return w_vector

    def get_ED_loss(self, batch_real_data):
        """
        Step I. Update E, and D:  Optimize the discriminator D(E( * )) to better differentiate between real x data ###
        and data generated by G(F( * ))
         """
        batch_z = torch.randn(batch_real_data.shape[0], self.latent_size, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            batch_fake_data = self.G(self.F(batch_z))
        fake_images_dicriminator_outputs = self.D(self.E(batch_fake_data))
        real_images_dicriminator_outputs = self.D(self.E(batch_real_data))
        loss = F.softplus(fake_images_dicriminator_outputs) + F.softplus(-real_images_dicriminator_outputs)

        r1_penalty = compute_r1_gradient_penalty(real_images_dicriminator_outputs, batch_real_data)

        loss += self.hp['g_penalty_coeff'] * r1_penalty
        loss = loss.mean()

        return loss

    def get_FG_loss(self, batch_real_data):
        batch_z = torch.randn(batch_real_data.shape[0], self.latent_size, dtype=torch.float32).to(self.device)
        batch_fake_data = self.G(self.F(batch_z))
        fake_images_dicriminator_outputs = self.D(self.E(batch_fake_data))
        loss = F.softplus(-fake_images_dicriminator_outputs).mean()

        return loss

    def get_EG_loss(self, batch_real_data):
        batch_z = torch.randn(batch_real_data.shape[0], self.latent_size, dtype=torch.float32).to(self.device)
        # with torch.no_grad():
        batch_w = self.F(batch_z)
        batch_reconstructed_w = self.E(self.G(batch_w))
        # return F.mse_loss(batch_w.detach(), batch_reconstructed_w)
        return torch.mean(((batch_reconstructed_w - batch_w.detach())**2))

    def train(self, train_dataloader, test_data, epohcs, output_dir):
        tracker = LossTracker(output_dir)
        for epoch in range(epohcs):
            for batch_real_data in tqdm(train_dataloader):
                # Step I. Update E, and D: optimizer the discriminator D(E( * ))
                self.ED_optimizer.zero_grad()
                L_adv_ED = self.get_ED_loss(batch_real_data)
                L_adv_ED.backward()
                self.ED_optimizer.step()
                tracker.update(dict(L_adv_ED=L_adv_ED))

                # Step II. Update F, and G: Optimize the generator G(F( * )) to fool D(E ( * ))
                self.FG_optimizer.zero_grad()
                L_adv_FG = self.get_FG_loss(batch_real_data)
                L_adv_FG.backward()
                self.FG_optimizer.step()
                tracker.update(dict(L_adv_FG=L_adv_FG))

                # Step III. Update E, and G: Optimize the reconstruction loss in the Latent space W
                # self.ED_optimizer.zero_grad()
                # self.FG_optimizer.zero_grad()
                self.EG_optimizer.zero_grad()
                L_err_EG = self.get_EG_loss(batch_real_data)
                L_err_EG.backward()
                self.EG_optimizer.step()
                # self.ED_optimizer.step()
                # self.FG_optimizer.step()
                tracker.update(dict(L_err_EG=L_err_EG))

            self.save_sample(epoch, tracker, test_data[1], test_data[0], output_dir)

    def save_sample(self, epoch, tracker, samples, samples_z, output_dir):
        with torch.no_grad():
            restored_image = self.generate(self.encode(samples))
            generated_images = self.generate(samples_z)

            # tracker.update(dict(gen_min_val=0.5 * (restored_image.min() + generated_images.min()),
            #                     gen_max_val=0.5 * (restored_image.max() + generated_images.max())))

            resultsample = torch.cat([samples, restored_image, generated_images], dim=0).cpu()

            # Normalize images from -1,1 to 0, 1.
            # Eventhough train samples are in this range (-1,1), the generated image may not. But this should diminish as
            # raining continues or else the discriminator can detect them. Anyway save_image clamps it to 0,1
            resultsample = resultsample * 0.5 + 0.5

            tracker.register_means(epoch)
            tracker.plot()
            f = os.path.join(output_dir, 'sample_%d.jpg' % (epoch))
            save_image(resultsample, f, nrow=len(samples))