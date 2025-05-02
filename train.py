import torch
import numpy as np
import copy
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from unet import UNet
from carvana_dataset import CarvanaDataset
from matplotlib import pyplot as plt


if __name__ == "__main__":
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 2
    EPOCHS = 20
    DATA_PATH = r'C:\Users\joyvi\Desktop\inmemcomp\data'
    MODEL_SAVE_PATH = r'C:\Users\joyvi\Desktop\inmemcomp\unet_carvana.pth'
    tr_loss_lst = []
    tst_loss_lst = []
    val_loss_lst = []
    tr_dice = []
    val_dice = []
    test_dice = []
    tr_iou = []
    val_iou = []
    test_iou = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CarvanaDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset , test_dataset = random_split(train_dataset, [0.8,0.15,0.05], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        mdice_train = 0
        mIoU = 0
        cnt = 0 
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            cnt += 1
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            intersection = torch.sum((y_pred > 0.5).int() * (mask > 0.5).int())
            sm = torch.sum((y_pred > 0.5).int()) + torch.sum((mask > 0.5).int())
            union = torch.sum((y_pred > 0.5).int() + (mask > 0.5).int() - (y_pred > 0.5).int() * (mask > 0.5).int())
            mdice_train += 2*intersection/sm
            mIoU += intersection/union
            # print("intersection : " , intersection)
            # print("union : " , union)
            # print("dice : " , 2*intersection/union)
        tr_dice.append(mdice_train.item()/cnt)
        tr_iou.append(mIoU.item()/cnt)
        train_loss = train_running_loss / cnt
        #print("m dice : " , tr_dice)
        #print("BCE : " , train_loss)
        
        model.eval()
        val_running_loss = 0
        mdice_val = 0
        cnt = 0
        mIoU = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                cnt += 1
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

                intersection = torch.sum((y_pred > 0.5).int() * (mask > 0.5).int())
                sm = torch.sum((y_pred > 0.5).int()) + torch.sum((mask > 0.5).int())
                union = torch.sum((y_pred > 0.5).int() + (mask > 0.5).int() - (y_pred > 0.5).int() * (mask > 0.5).int())
                mdice_val += 2*intersection/sm
                mIoU += intersection/union
                
                

            val_dice.append(mdice_val.item()/cnt)  
            val_iou.append(mIoU.item()/cnt)  
            val_loss = val_running_loss /cnt
        test_running_loss = 0
        mdice_test = 0
        cnt = 0
        mIoU = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(test_dataloader)):
                cnt += 1
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)

                test_running_loss += loss.item()
                
                intersection = torch.sum((y_pred > 0.5).int() * (mask > 0.5).int())
                sm = torch.sum((y_pred > 0.5).int()) + torch.sum((mask > 0.5).int())
                union = torch.sum((y_pred > 0.5).int() + (mask > 0.5).int() - (y_pred > 0.5).int() * (mask > 0.5).int())
                mdice_test += 2*intersection/sm
                mIoU += intersection/union

            test_dice.append(mdice_test.item()/cnt)  
            test_iou.append(mIoU.item()/cnt)
            test_loss = test_running_loss /cnt    

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print(f"Test Loss EPOCH {epoch+1}: {test_loss:.4f}")
        print(f"Train dice EPOCH {epoch+1}: {tr_dice[-1]:.4f}")
        print(f"Valid dice EPOCH {epoch+1}: {val_dice[-1]:.4f}")
        print(f"Test dice EPOCH {epoch+1}: {test_dice[-1]:.4f}")
        print(f"Train iou EPOCH {epoch+1}: {tr_iou[-1]:.4f}")
        print(f"Valid iou EPOCH {epoch+1}: {val_iou[-1]:.4f}")
        print(f"Test iou EPOCH {epoch+1}: {test_iou[-1]:.4f}")
        tr_loss_lst.append(train_loss)
        tst_loss_lst.append(test_loss)
        val_loss_lst.append(val_loss)
        print("-"*30)

    epoch = [i+1 for i in range(EPOCHS)]
    # tr_loss_lst = epoch*4
    # val_loss_lst = epoch*2.5
    # tst_loss_lst = epoch*1.2
    # tr_dice = epoch * 2
    # test_dice = epoch * 4
    # val_dice = epoch * 0.5
    # tr_iou = epoch * 7
    # test_iou = epoch * 3
    # val_iou = epoch * 5
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].plot(epoch , tr_loss_lst , label = 'Training loss')
    axes[0].plot(epoch , val_loss_lst , label = 'Validation loss')
    axes[0].plot(epoch , tst_loss_lst , label = 'Test loss')
    axes[0].scatter(epoch , tr_loss_lst)
    axes[0].scatter(epoch , val_loss_lst)
    axes[0].scatter(epoch , tst_loss_lst)
    axes[0].set_title("BCE loss vs epoch")
    axes[0].set_xlabel("Epochs") 
    axes[0].set_ylabel("Loss")  
    axes[0].legend()
    axes[0].grid()  
    axes[1].plot(epoch , tr_dice , label = 'Dice score - train')
    axes[1].plot(epoch , val_dice  , label = 'Dice score - validation')
    axes[1].plot(epoch , test_dice , label = 'Dice score - test')
    axes[1].scatter(epoch , tr_dice)
    axes[1].scatter(epoch , val_dice)
    axes[1].scatter(epoch , test_dice)
    axes[1].set_title("mean dice vs epoch")
    axes[1].set_xlabel("Epochs") 
    axes[1].set_ylabel("Dice") 
    axes[1].legend() 
    axes[1].grid()
    axes[2].plot(epoch , tr_dice , label = 'mIoU - train')
    axes[2].plot(epoch , val_dice  , label = 'mIoU - validation')
    axes[2].plot(epoch , test_dice , label = 'mIoU  - test')
    axes[2].scatter(epoch , tr_dice)
    axes[2].scatter(epoch , val_dice)
    axes[2].scatter(epoch , test_dice)
    axes[2].set_title("mean IoU vs epoch")
    axes[2].set_xlabel("Epochs") 
    axes[2].set_ylabel("mIoU") 
    axes[2].legend() 
    axes[2].grid()
    plt.show()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)   