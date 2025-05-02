# U-net
This repository is an implementation of U-Net architecture for binary segmentation of car images from caravana dataset. The architecture of the U-Net is as depicted below :

![image alt](https://github.com/mr-unknown-oo7/U-net/blob/main/unet.png?raw=true)


The neural networks was trained for 20 epochs and a batch size of 2 with a learning rate of 5e-4 achieving a BCE loss of 0.0105 , mean IoU of 0.9842 and a mean Dice score of 0.9920 in the test dataset. The plots of mean dice vs epoch , mean IoU vs epoch and BCE loss vs epoch are as shown below 

![image alt2](https://github.com/mr-unknown-oo7/U-net/blob/main/plot-unet.png?raw=true)
Further the trained model was tested no 2 random data achiveing the following results 

![img alt3](https://github.com/mr-unknown-oo7/U-net/blob/main/u%20net%20img%202.png?raw=true)
![img alt4](https://github.com/mr-unknown-oo7/U-net/blob/main/unet%20img%201.png?raw=true)

# Future Prospects 
The project would be extended to its Spiking Version soon.
