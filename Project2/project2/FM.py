import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionalFM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels,
            self.modelconfig.num_feat,
            self.modelconfig.num_classes,
            self.modelconfig.input_dim,
        )
        self.to(device)

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
        t = torch.rand(B, device=device).view(B, 1, 1, 1)  # t in [0,1]

        c = F.one_hot(conditions.long(), num_classes=self.modelconfig.num_classes).float().to(device)

        x0 = torch.randn_like(images, device=device)
        x1 = images
        xt = (1 - t) * x0 + t * x1

        t_input = t.view(B, 1, 1, 1)

        v_pred = self.network(xt, t_input, c)

        u_t = x1 - x0

        loss = self.loss_fn(v_pred, u_t)


        # ==================================================== #
        return loss

    def sample(self, conditions, omega):
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
        c_null = torch.zeros_like(c).to(device)

        x = torch.randn(B, self.modelconfig.num_channels, self.modelconfig.input_dim, self.modelconfig.input_dim, device=device)

        dt = 1.0 / self.modelconfig.T

        self.network.eval()
        with torch.no_grad():
            for i in range(self.modelconfig.T):
                t = i * dt
                t_input = torch.full((B, 1, 1, 1), t, device=device)

                v_cond = self.network(x, t_input, c)
                v_uncond = self.network(x, t_input, c_null)
                v_pred = (1 + omega) * v_cond - omega * v_uncond

                x = x + dt * v_pred

        self.network.train()


        # ==================================================== #
        generated_images = (x * 0.3081 + 0.1307).clamp(0, 1)
        return generated_images