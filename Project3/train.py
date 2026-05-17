"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig

def solver(model_name):
    # Initialize the model
    if model_name == "bigram":
        config = BigramConfig
        model = BigramLanguageModel(config)
    elif model_name == "minigpt":
        config = MiniGPTConfig
        model = MiniGPT(config)
    else:
        raise ValueError("Invalid model name")
    
    # Load the dataset
    train_dataset = TinyStoriesDataset(
        config.path_to_data,
        mode="train",
        context_length=config.context_length,
    )
    eval_dataset = TinyStoriesDataset(
        config.path_to_data, mode="test", context_length=config.context_length
    )

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.batch_size, pin_memory=True
    )

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print number of parameters in the model
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))

    # Initialize wandb if you want to use it
    if config.to_log:
        wandb.init(project="dl2_proj3")

    # Create the save path if it does not exist
    if not Path.exists(config.save_path):
        Path.mkdir(config.save_path, parents=True, exist_ok=True)

    ### ==================== START OF YOUR CODE ==================== ###
    """
    You are required to implement the training loop for the model.

    The code below is a skeleton for the training loop, for your reference. 
    You can fill in the missing parts or completely set it up from scratch.

    Please keep the following in mind:
    - You will need to define an appropriate loss function for the model.
    - You will need to define an optimizer for the model.
    - You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
    - It is recommended that you save the model weights every `config.save_iterations` iterations. You can also just save the model with the best training loss.

    NOTE : 
    - Please check the config file to see the different configurations you can set for the model.
    - The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
    not a required part of the assignment. 
    - Feel free to experiment with the parameters and I would be happy to talk to you about them if interested.
    """

    ### ========= TODO : START ========= ###
    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ### ======== TODO : END ========= ###

    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=2
        )

    model.train()
    model.to(device)

    best_train_loss = float("inf")
    eval_loss = None

    for i, (context, target) in enumerate(train_dataloader):

        train_loss = None # You can use this variable to store the training loss for the current iteration
        ### ======== TODO : START ========= ###
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(context)
        train_loss = loss(
            rearrange(logits, "b t v -> (b t) v"),
            rearrange(target, "b t -> (b t)"),
        )
        train_loss.backward()
        if config.to_clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        train_loss = train_loss.item()
        ### ======== TODO : END ========= ###

        if config.scheduler:
            scheduler.step()

        del context, target # Clear memory

        if i % config.log_interval == 0:

            model.eval()
            eval_loss = None # You can use this variable to store the evaluation loss for the current iteration
            ### ======== TODO : START ========= ###
            eval_losses = []
            max_eval_batches = 50
            with torch.no_grad():
                for j, (eval_context, eval_target) in enumerate(eval_dataloader):
                    eval_context = eval_context.to(device)
                    eval_target = eval_target.to(device)
                    eval_logits = model(eval_context)
                    batch_loss = loss(
                        rearrange(eval_logits, "b t v -> (b t) v"),
                        rearrange(eval_target, "b t -> (b t)"),
                    )
                    eval_losses.append(batch_loss.item())
                    if j >= max_eval_batches - 1:
                        break
            eval_loss = sum(eval_losses) / len(eval_losses)
            ### ======== TODO : END ========= ###
            
            print(
                f"Iteration {i}, Train Loss: {train_loss}",
                f"Eval Loss: {eval_loss}",
            )

            if config.to_log:
                wandb.log(
                    {
                        "Train Loss": train_loss,
                        "Eval Loss": eval_loss,
                    }
                )

            model.train()

        # Save the model every config.save_iterations
        if i % config.save_iterations == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "iteration": i,
                },
                config.save_path / f"mini_model_checkpoint_{i}.pt",
            )

        if i > config.max_iter:
            break