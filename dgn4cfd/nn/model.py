import contextlib
import os
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Union

from ..graph import Graph
from ..loader import DataLoader


class Model(nn.Module):
    r"""Base class for all the GNN models.

    Args:
        arch (Optional[Union[None,dict]], optional): Dictionary with the model architecture. Defaults to `None`.
        weights (Optional[Union[None,str]], optional): Path of the weights file. Defaults to `None`.
        model (Optional[Union[None,str]], optional): Path of the checkpoint file. Defaults to `None`.
        device (Optional[torch.device], optional): Device where the model is loaded. Defaults to `torch.device('cpu')`.
    """

    def __init__(
        self,
        arch: dict = None,
        weights: str = None,
        checkpoint: str = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.load_model(arch, weights, checkpoint)

    # To be overwritten
    @property
    def num_fields(self) -> int:
        """Returns the number of output fields. It must be overloaded by each model instancing the `graphs4cfd.models.Model` class."""
        pass

    def load_model(self, arch, weights, checkpoint):
        """Loads the model architecture from a arch dictionary and its weights from a weights file, or loads the model from a checkpoint file."""
        if arch is not None and checkpoint is None:
            self.load_arch(arch)
            # To device
            self.to(self.device)
            if weights is not None:
                self.load_state_dict(
                    torch.load(weights, map_location=self.device, weights_only=True)
                )

        elif arch is None and weights is None and checkpoint is not None:
            checkpoint = torch.load(
                checkpoint, map_location=self.device, weights_only=True
            )
            if "diffusion_process" in checkpoint.keys():
                self.diffusion_process = checkpoint["diffusion_process"]
            if "learnable_variance" in checkpoint.keys():
                self.learnable_variance = checkpoint["learnable_variance"]
            self.load_arch(checkpoint["arch"])
            self.to(self.device)
            self.load_state_dict(checkpoint["weights"])
        return

    # To be overwritten
    def load_arch(self, arch: dict):
        """Defines the hyper-parameters of the model. It must be overloaded by each model instancing the `graphs4cfd.models.Model` class.

        Args:
            arch (dict): Dictionary with the architecture of the model. Its structure depends on the model.
        """
        pass

    # To be overwritten
    def forward(self, graph: Graph):
        """Forwrad pass (or time step) of the model. It must be overloaded by each model instancating the `graphs4cfd.models.Model` class.

        Args:
            graph (Graph): Graph object with the input data.
            t (int): current time-point
        """
        pass

    def fit(
        self,
        trainining_settings: object,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        """Trains the model.

        Args:
            config       (TrainingSettings):               Configuration of the training.
            train_loader (DataLoader):                     Training data loader.
            val_loader   (Optional[DataLoader], optional): Validation data loader. Defaults to `None`.
        """
        # Change the training device if needed
        if (
            trainining_settings["device"] is not None
            and trainining_settings["device"] != self.device
        ):
            self.to(trainining_settings["device"])
            self.device = trainining_settings["device"]
        # Set the training loss
        criterion = trainining_settings["training_loss"]
        # Load checkpoint
        checkpoint = None
        scheduler = None
        if trainining_settings["checkpoint"] is not None and os.path.exists(
            trainining_settings["checkpoint"]
        ):
            print(
                "Training from an existing checkpoint:",
                trainining_settings["checkpoint"],
            )
            checkpoint = torch.load(
                trainining_settings["checkpoint"],
                map_location=self.device,
                weights_only=True,
            )
            self.load_state_dict(checkpoint["weights"])
            optimiser = torch.optim.Adam(self.parameters(), lr=checkpoint["lr"])
            optimiser.load_state_dict(checkpoint["optimiser"])
            if trainining_settings["scheduler"] is not None:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser,
                    factor=trainining_settings["scheduler"]["factor"],
                    patience=trainining_settings["scheduler"]["patience"],
                    eps=0.0,
                )
                scheduler.load_state_dict(checkpoint["scheduler"])
            initial_epoch = checkpoint["epoch"] + 1
        # Initialise optimiser and scheduler if not previous checkpoint is used
        else:
            # If a .chk is given but it does not exist such file, notify the user
            if trainining_settings["checkpoint"] is not None:
                print(
                    "Not matching checkpoint file:", trainining_settings["checkpoint"]
                )
            print("Training from randomly initialised weights")
            optimiser = optim.Adam(self.parameters(), lr=trainining_settings["lr"])
            if trainining_settings["scheduler"] is not None:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser,
                    factor=trainining_settings["scheduler"]["factor"],
                    patience=trainining_settings["scheduler"]["patience"],
                    eps=0.0,
                )
            initial_epoch = 1
        # If .chk to save exists rename the old version to .bck
        path = os.path.join(
            trainining_settings["folder"], trainining_settings["name"] + ".chk"
        )
        if os.path.exists(path):
            print("Renaming", path, "to:", path + ".bck")
            os.rename(path, path + ".bck")
        # Initialise tensor board writer
        if trainining_settings["tensor_board"] is not None:
            writer = SummaryWriter(
                os.path.join(
                    trainining_settings["tensor_board"], trainining_settings["name"]
                )
            )
        # Initialise automatic mixed-precision training
        scaler = None
        if trainining_settings["mixed_precision"]:
            print("Training with automatic mixed-precision")
            scaler = torch.cuda.amp.GradScaler()
            # Load previos scaler
            if checkpoint is not None and checkpoint["scaler"] is not None:
                scaler.load_state_dict(checkpoint["scaler"])
        # Print before training
        print(f"Training on device: {self.device}")
        print(f"Number of learnable parameters: {self.num_learnable_params}")
        print(f"Total number of parameters:     {self.num_params}")
        # Training loop
        for epoch in tqdm(
            range(initial_epoch, trainining_settings["epochs"] + 1),
            desc="Completed epochs",
            leave=False,
            position=0,
        ):
            if optimiser.param_groups[0]["lr"] < trainining_settings["stopping"]:
                print(
                    f"The learning rate is smaller than {trainining_settings['stopping']}. Stopping training."
                )
                self.save_checkpoint(
                    path, epoch, optimiser, scheduler=scheduler, scaler=scaler
                )
                break
            print("\n")
            print(f"Hyperparameters: lr = {optimiser.param_groups[0]['lr']}")
            self.train()
            training_loss = 0.0
            gradients_norm = 0.0
            for iteration, data in enumerate(train_loader):
                data = data.to(self.device)
                with (
                    torch.cuda.amp.autocast()
                    if trainining_settings["mixed_precision"]
                    else contextlib.nullcontext()
                ):  # Use automatic mixed-precision
                    loss = criterion(self, data)
                # Back-propagation
                if trainining_settings["mixed_precision"]:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                # Save training loss and gradients norm before applying gradient clipping to the weights
                training_loss += loss.item()
                gradients_norm += self.grad_norm()
                # Update the weights
                if trainining_settings["mixed_precision"]:
                    # Clip the gradients
                    if (
                        trainining_settings["grad_clip"] is not None
                        and epoch > trainining_settings["grad_clip"]["epoch"]
                    ):
                        scaler.unscale_(optimiser)
                        nn.utils.clip_grad_norm_(
                            self.parameters(), trainining_settings["grad_clip"]["limit"]
                        )
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    # Clip the gradients
                    if (
                        trainining_settings["grad_clip"] is not None
                        and epoch > trainining_settings["grad_clip"]["epoch"]
                    ):
                        nn.utils.clip_grad_norm_(
                            self.parameters(), trainining_settings["grad_clip"]["limit"]
                        )
                    optimiser.step()
                # Reset the gradients
                optimiser.zero_grad()
            training_loss /= iteration + 1
            gradients_norm /= iteration + 1
            # Display on terminal
            print(
                f"Epoch: {epoch:4d}, Training   loss: {training_loss:.4e}, Gradients: {gradients_norm:.4e}"
            )
            # Testing
            if val_loader is not None:
                validation_criterion = trainining_settings["validation_loss"]
                self.eval()
                with torch.no_grad():
                    validation_loss = 0.0
                    for iteration, data in enumerate(val_loader):
                        data = data.to(self.device)
                        validation_loss += validation_criterion(self, data).item()
                    validation_loss /= iteration + 1
                    print(f"Epoch: {epoch:4d}, Validation loss: {validation_loss:.4e}")
            # Log in TensorBoard
            if trainining_settings["tensor_board"] is not None:
                writer.add_scalar("Loss/train", training_loss, epoch)
                if val_loader:
                    writer.add_scalar("Loss/test", validation_loss, epoch)
            # Update lr
            if trainining_settings["scheduler"]["loss"][:2] == "tr":
                scheduler_loss = training_loss
            elif trainining_settings["scheduler"]["loss"][:3] == "val":
                scheduler_loss = validation_loss
            scheduler.step(scheduler_loss)
            # Create training checkpoint
            if not epoch % trainining_settings["chk_interval"]:
                print("Saving checkpoint in:", path)
                self.save_checkpoint(
                    path, epoch, optimiser, scheduler=scheduler, scaler=scaler
                )
        # Save final checkpoint
        self.save_checkpoint(path, epoch, optimiser, scheduler=scheduler, scaler=scaler)
        writer.close()
        print("Finished training")
        return

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        optimiser: torch.optim.Optimizer,
        scheduler: Union[None, dict] = None,
        scaler: Union[None, dict] = None,
    ) -> None:
        """Saves the model parameters, the optimiser state and the current value of the training hyper-parameters.
        The saved file can be used to resume training with the `graphs4cfd.nn.model.Model.fit` method."""
        checkpoint = {
            "weights": self.state_dict(),
            "optimiser": optimiser.state_dict(),
            "lr": optimiser.param_groups[0]["lr"],
            "epoch": epoch,
        }
        if hasattr(self, "arch"):
            checkpoint["arch"] = self.arch
        if hasattr(self, "diffusion_process"):
            checkpoint["diffusion_process"] = self.diffusion_process
        if hasattr(self, "learnable_variance"):
            checkpoint["learnable_variance"] = self.learnable_variance
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
        torch.save(checkpoint, filename)
        return

    @property
    def num_params(self):
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_learnable_params(self):
        """Returns the number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def grad_norm(self):
        """Returns the L2 norm of the gradients."""
        norm = 0.0
        for p in self.parameters():
            if p.requires_grad:
                norm += p.grad.data.norm(2).item() ** 2
        return norm**0.5
