import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os 

from carvana_dataset import CarvanaDataset
from unet import UNet


def single_image_inference(imge, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    n = len(imge)
    cnt = 1 
    for i in imge:
      transform = transforms.Compose([transforms.Resize((512, 512)),transforms.ToTensor()])
      img = transform(Image.open(i)).float().to(device)
      img = img.unsqueeze(0)
    
      pred_mask = model(img)

      img = img.squeeze(0).cpu().detach()
      img = img.permute(1, 2, 0)

      pred_mask = pred_mask.squeeze(0).cpu().detach()
      pred_mask = pred_mask.permute(1, 2, 0)
      pred_mask[pred_mask < 0]=0
      pred_mask[pred_mask > 0]=1

      fig = plt.figure()
      for j in range(0, 2): 
          fig.add_subplot(4, 2, cnt + j)
          print(cnt+j)
          if j%2 == 1:
              plt.imshow(img, cmap="gray")
          else:
              plt.imshow(pred_mask, cmap="gray")
      cnt += 2
    plt.show()


if __name__ == "__main__":
    SINGLE_IMG_PATH = './test/'
    MODEL_PATH = r'C:\Users\joyvi\Desktop\inmemcomp\unet_carvana.pth'
    images = sorted([SINGLE_IMG_PATH+i for i in os.listdir(SINGLE_IMG_PATH)])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(images, MODEL_PATH, device)