import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels, 
            self.modelconfig.num_feat, 
            self.modelconfig.num_classes, 
            self.modelconfig.input_dim
        )
        self.to(device)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.modelconfig.beta_1, self.modelconfig.beta_T, self.modelconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       

        beta_t = (beta_1 + (t_s.float() - 1) / (T - 1) * (beta_T - beta_1)).reshape(-1, 1)  # no .to(device)
        sqrt_beta_t = torch.sqrt(beta_t)

        alpha_t = (1.0 - beta_t).reshape(-1, 1)
        betas_full = beta_1 + (torch.arange(1, T + 1, device=t_s.device).float() - 1) / (T - 1) * (beta_T - beta_1)
        alphas_full = 1.0 - betas_full
        alpha_bar_full = torch.cumprod(alphas_full, dim=0)  # (T,)

        t_idx = (t_s.long() - 1).clamp(0, T - 1).reshape(-1)  
        alpha_t_bar = alpha_bar_full[t_idx].reshape(-1, 1)        

        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1.0 - alpha_t_bar)
        oneover_sqrt_alpha = 1.0 / torch.sqrt(alpha_t)


        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function.  



        B = images.shape[0]
        t_s = torch.randint(1, self.modelconfig.T + 1, (B, 1), device=device).float()

        c = F.one_hot(conditions.long(), num_classes=self.modelconfig.num_classes).float().to(device)

        drop_mask = (torch.rand(B, 1, device=device) < self.modelconfig.mask_p)
        c_null = torch.full_like(c, float(self.modelconfig.condition_mask_value))
        c = torch.where(drop_mask, c_null, c)

        sched = self.scheduler(t_s)
        sqrt_alpha_bar       = sched['sqrt_alpha_bar'].view(B, 1, 1, 1)
        sqrt_oneminus_alpha_bar = sched['sqrt_oneminus_alpha_bar'].view(B, 1, 1, 1)

        noise = torch.randn_like(images, device=device)
        x_t = sqrt_alpha_bar * images + sqrt_oneminus_alpha_bar * noise

        t_input = (t_s / self.modelconfig.T).view(B, 1, 1, 1).to(device)

        noise_pred = self.network(x_t, t_input, c)

        noise_loss = self.loss_fn(noise_pred, noise)


        # ==================================================== #
        return noise_loss

    def sample(self, conditions, omega):
        T = self.modelconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  


        B = conditions.shape[0]

        c = F.one_hot(conditions.long(), num_classes=self.modelconfig.num_classes).float().to(device)
        c_null = torch.full_like(c, float(self.modelconfig.condition_mask_value))
        X_t = torch.randn(B, self.modelconfig.num_channels,
                        self.modelconfig.input_dim, self.modelconfig.input_dim, device=device)

        self.network.eval()
        with torch.no_grad():
            for t in tqdm(range(T, 0, -1), desc="Sampling"):
                t_s = torch.full((B, 1), t, dtype=torch.float32, device=device)
                t_input = (t_s / T).view(B, 1, 1, 1)

                sched = self.scheduler(t_s)
                beta_t                  = sched['beta_t'].view(B, 1, 1, 1)
                sqrt_oneminus_alpha_bar = sched['sqrt_oneminus_alpha_bar'].view(B, 1, 1, 1)
                oneover_sqrt_alpha      = sched['oneover_sqrt_alpha'].view(B, 1, 1, 1)
                sqrt_beta_t             = sched['sqrt_beta_t'].view(B, 1, 1, 1)

                noise_cond   = self.network(X_t, t_input, c)
                noise_uncond = self.network(X_t, t_input, c_null)
                noise_pred   = (1 + omega) * noise_cond - omega * noise_uncond

                z = torch.randn_like(X_t) if t > 1 else torch.zeros_like(X_t)
                X_t = oneover_sqrt_alpha * (X_t - (beta_t / sqrt_oneminus_alpha_bar) * noise_pred) + sqrt_beta_t * z

        self.network.train()


        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1)
        return generated_images