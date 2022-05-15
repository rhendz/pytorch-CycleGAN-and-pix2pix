import numpy as np
import matplotlib.pyplot as plt

def plot_vae_image(encoder, decoder, train_loader):
    for data in train_loader:

      real_img = data['A'][0]
      plt.imshow(real_img.permute(1, 2, 0))
      plt.show()
      gen_img = decoder(encoder(data['A']))
      print(gen_img[0].cpu().size())
      plt.imshow(gen_img[0].cpu().permute(1, 2, 0).detach().numpy())
      plt.show()
      break;