import torch
import torch.nn as nn
import torch.utils.data


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))
    
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        
        # Flatten the input tensor and compute its length
        self.input_size = 256 * 12 * 17 + 2
        self.output_size = 256 * 12 * 17

        # Define layers
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 4096),  # Example size for the hidden layer
            nn.ReLU(),
            nn.Linear(4096, self.output_size)
        )

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        
        # Feed-forward
        x = self.layers(x)
        
        # Reshape back to the desired output shape
        x = x.view(x.size(0), 256, 12, 17)
        return x


class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n

    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """
    def __init__(self, channels, ch=64, latent_channels=512):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, ch, 7, 1, 3)
        self.res_down_block1 = ResDown(ch, 2 * ch)
        self.res_down_block2 = ResDown(2 * ch, 4 * ch)
        self.res_down_block3 = ResDown(4 * ch, 8 * ch)
        self.res_down_block4 = ResDown(8 * ch, 16 * ch)
        self.conv_mu = nn.Conv2d(16 * ch, latent_channels, 4, 1)
        self.conv_log_var = nn.Conv2d(16 * ch, latent_channels, 4, 1)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        x = self.act_fnc(self.conv_in(x))
        x = self.res_down_block1(x)  # 32
        x = self.res_down_block2(x)  # 16
        x = self.res_down_block3(x)  # 8
        x = self.res_down_block4(x)  # 4
        mu = self.conv_mu(x)  # 1
        log_var = self.conv_log_var(x)  # 1

        if self.training:
            x = self.sample(mu, log_var)
        else:
            x = mu

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, latent_channels=512):
        super(Decoder, self).__init__()
        self.conv_t_up = nn.ConvTranspose2d(latent_channels, ch * 16, 4, 1)
        self.res_up_block1 = ResUp(ch * 16, ch * 8)
        self.res_up_block2 = ResUp(ch * 8, ch * 4)
        self.res_up_block3 = ResUp(ch * 4, ch * 2)
        self.res_up_block4 = ResUp(ch * 2, ch)
        self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_t_up(x))  # 4
        x = self.res_up_block1(x)  # 8
        x = self.res_up_block2(x)  # 16
        x = self.res_up_block3(x)  # 32
        x = self.res_up_block4(x)  # 64
        x = torch.tanh(self.conv_out(x))

        return x 


class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, channel_in=3, ch=64, latent_channels=512, saccade_dim=2):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        
        self.encoder = Encoder(channel_in, ch=ch, latent_channels=latent_channels)
        self.decoder = Decoder(channel_in, ch=ch, latent_channels=latent_channels)

    def forward(self, x, saccade_vectors):
        print(x.size())
        
        encoding, mu, log_var = self.encoder(x)

        reshaped_encoding = encoding.view(encoding.size(0), -1)

        saccade_vectors = saccade_vectors.cuda()
        saccade_vectors_encoding = saccade_vectors.view(encoding.size(0), -1)

        print(saccade_vectors_encoding.shape)
        print(reshaped_encoding.shape) 

        # Reshape the second tensor
        
        saccade_reshaped = saccade_vectors.unsqueeze(-1).unsqueeze(-1)
        
        # Expand the dimensions of the reshaped tensor to match those of tensor_4d
        saccade_expanded = saccade_reshaped.expand(-1, -1, 12, 17)
        
        # Concatenate along the channel dimension (dim=1)
        result = torch.cat((encoding, saccade_expanded), dim=1)

        result2 = torch.cat((reshaped_encoding, saccade_vectors_encoding), dim=1)

        print(result2.shape)

        model = FeedForwardNet().cuda()
        
        # Reshape the new latent back to the original mu shape
        new_latent = model(result2)
        
        # Pass the new latent vector through the decoder
        recon_img = self.decoder(new_latent)

        return recon_img, mu, log_var

