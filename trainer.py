import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from numpy.typing import NDArray
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

from dataset import GraspDataset


class MobileUNet(nn.Module):
    def __init__(self, pretrained: bool=True, num_rots: int=2) -> None:
        '''Implements UNet style network with pretrained MobileNetv3 backbone

        Refer to the provided diagram for details on the layers.  We have already
        provided the code needed to use the MobileNet layers: self.backbone1 is the
        first part of the MobileNet that produces a 24-channel, 8x8 feature map;
        self.backbone2 produces a 40-channel, 4x4 feature map.

        You do not need to include any batch norm layers.  All conv1x1 layers
        have stride=1 and padding=0.  To perform upsampling of the feature maps,
        we recommend using `nn.Upsample` and setting the argument
        `align_corners=True`.

        The two pathways in the network are combined using addition. The output
        of the network should not have an activation function applied to it
        '''
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        all_layers = mobilenet_v3_small(weights=weights).features

        # output of self.backbone1 is tensor of shape (B, 24, 8, 8)
        self.backbone1 = nn.Sequential(*[all_layers[i] for i in range(3)])

        # output of self.backbone2 is tensor of shape (B, 40, 4, 4)
        self.backbone2 = nn.Sequential(*[all_layers[i] for i in range(3, 6)])

        ##########################
        # define other layers here
        ##########################
        self.up_1 = nn.Sequential(
            nn.Conv2d(40, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.up_2 = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0)
        )
        self.up_3 = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.up_4 = nn.Sequential(
            nn.Conv2d(24, 4, kernel_size=1, stride=1, padding=0)
        )


    def forward(self, img: Tensor) -> Tensor:
        '''Perform forward pass to generate logits over pixel location and gripper
        rotation

        Arguments
        ---------
        img: float tensor of shape (B, 3, H, W)

        Returns
        -------
        float tensor of logits with shape (B, 2, H, W)
        '''
        # The processing in the MobileNet is already done here
        x_wide = self.backbone1(img) # tensor of shape (B, 24, 8, 8)
        x_narrow = self.backbone2(x_wide) # tensor of shape (B, 40, 4, 4)

        ############################################
        # implement additional steps in forward pass
        ############################################s
        # Upsample and convolutions
        x_up_1 = self.up_1(x_narrow)  # tensor of shape (B, 128, 4, 4)
        x_up_1 = F.interpolate(x_up_1, scale_factor=8, mode='bilinear',
                               align_corners=True)  # tensor of shape (B, 128, 32, 32)

        x_up_2 = self.up_2(x_up_1)  # tensor of shape (B, 4, 32, 32)

        x_up_3 = self.up_3(x_up_2)  # tensor of shape (B, 2, 32, 32)
        x_up_3 = F.interpolate(x_up_3, scale_factor=2, mode='bilinear',
                               align_corners=True)  # tensor of shape (B, 2, 64, 64)

        x_up_4 = F.interpolate(x_wide, scale_factor=4, mode='bilinear',
                               align_corners=True)  # tensor of shape (B, 24, 32, 32)
        x_up_4 = self.up_4(x_up_4)  # tensor of shape (B, 4, 32, 32)

        x_combined = x_up_2 + x_up_4  # tensor of shape (B, 4, 32, 32)

        logits = F.interpolate(x_combined, scale_factor=2, mode='bilinear',
                               align_corners=True)  # tensor of shape (B, 2, 64, 64)

        return logits


    @torch.no_grad()
    def predict(self, img: Tensor) -> NDArray:
        '''Perform argmax over model output to find rot_id, px, py where maximum logit
        is achieved

        Arguments
        ---------
        img: Float tensor of shape (C, H, W)

        Returns
        -------
        array containing (rot_id, px, py)
        '''
        # perform forward pass and strip batch dim
        pred_logits = self.forward(img.unsqueeze(0)).squeeze(0)
        shape = pred_logits.shape

        # take argmax over pixel location and gripper rotation all at once
        max_id = torch.argmax(pred_logits.view(-1))

        # determine index of maximum along each dimension by unraveling
        pred_actions = np.unravel_index(max_id, shape)

        return pred_actions

    def save(self, path: str) -> None:
        '''Save model weights to *.pt file'''
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        '''Load model weights from *.pt file'''
        self.load_state_dict(torch.load(path))


def main():
    '''Instantiates and trains MobileUnet model to predict grasp success distribution
    over pixel location and gripper rotations.

    The training set has 800 successful grasps, and the validation set has 200.
    '''
    np.random.seed(42)
    torch.manual_seed(42)
    g = torch.Generator()
    g.manual_seed(42)

    # create dataloaders
    train_dataset = GraspDataset(train=True)
    val_dataset = GraspDataset(train=False)

    train_dl = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, generator=g,
    )
    val_dl = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=4, drop_last=True, generator=g,
    )

    model = MobileUNet(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=LR)

    pbar = tqdm(range(1, NUM_EPOCHS+1))
    record = dict(train_loss=[], val_loss=[])
    for epoch_num in pbar:

        train_losses = []
        model.train()
        for batch in train_dl:
            img, label = map(lambda x: x.to(DEVICE), batch)

            pred_logits = model.forward(img)
            loss = criterion(torch.flatten(pred_logits, 1), label)
            train_losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        val_losses = []
        model.eval()
        for batch in val_dl:
            img, label = map(lambda x: x.to(DEVICE), batch)

            with torch.no_grad():
                pred_logits = model.forward(img)
                loss = criterion(torch.flatten(pred_logits, 1), label)

            val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        record['train_loss'].append(avg_train_loss)
        record['val_loss'].append(avg_val_loss)

        # record checkpoint if val loss is lowest
        if avg_val_loss == np.min(record['val_loss']):
            model.save('grasp_mobilenet.pt')

        # logging details
        pbar.set_description(f'loss={avg_val_loss:.2f}')

        # plot learning curves
        f, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].plot(record['train_loss'])
        ax[0].set_title('Train Loss')
        ax[1].plot(record['val_loss'])
        ax[1].set_title('Val. Loss')
        plt.savefig('loss_curves.png')
        plt.close()

        if epoch_num % PLOT_FREQ == 0:
            f, ax = plt.subplots(5, 3, figsize=(5, 7))
            ax[0, 0].set_title(f'Input', fontsize=12)
            ax[0, 1].set_title(f'Logits (vert.)', fontsize=12)
            ax[0, 2].set_title(f'Logits (horiz.)', fontsize=12)
            for row in range(ax.shape[0]):
                ax[row, 0].imshow(img[row].permute(1, 2, 0).cpu())
                pred_probs = torch.softmax(
                    pred_logits[row].view(-1), -1,
                ).reshape(pred_logits.shape[1:])
                vmin, vmax = pred_probs.min().item(), pred_probs.max().item()
                ax[row, 1].imshow(pred_probs[0].cpu(), vmin=vmin, vmax=vmax)
                ax[row, 2].imshow(pred_probs[1].cpu(), vmin=vmin, vmax=vmax)

            [a.set_xticklabels([]) for a in ax.flatten()]
            [a.set_yticklabels([]) for a in ax.flatten()]
            plt.suptitle(f'Epoch = {epoch_num}')
            plt.savefig('predictions.png')
            plt.close()


if __name__ == "__main__":
    NUM_EPOCHS = 30
    BATCH_SIZE = 12
    LR = 5e-4
    PLOT_FREQ = 5 # plot predictions every this many number of epochs
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
