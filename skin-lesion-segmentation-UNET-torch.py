import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.transforms.functional as TF

def plot_all_metrics(history, save_dir='./'):
    epochs = range(1, len(history['epoch']) + 1)

    # Helper 函式畫單張圖
    def plot_curve(y_values, ylabel, title, filename, color='blue'):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, y_values, label=ylabel, color=color)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{filename}')
        plt.close()

    # Loss (Train & Val) 一張圖
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/loss_curve.png')
    plt.close()

    # 其他指標單獨一張張圖
    if 'val_iou' in history:
        plot_curve(history['val_iou'], 'IOU', 'Validation IOU', 'val_iou_curve.png', 'orange')
    if 'val_dice' in history:
        plot_curve(history['val_dice'], 'Dice', 'Validation Dice', 'val_dice_curve.png', 'green')
    if 'val_precision' in history:
        plot_curve(history['val_precision'], 'Precision', 'Validation Precision', 'val_precision_curve.png', 'purple')
    if 'val_recall' in history:
        plot_curve(history['val_recall'], 'Recall', 'Validation Recall', 'val_recall_curve.png', 'brown')
    if 'val_accuracy' in history:
        plot_curve(history['val_accuracy'], 'Accuracy', 'Validation Accuracy', 'val_accuracy_curve.png', 'cyan')

# dataset
class LesionDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform_img=None, transform_mask=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.augment = augment


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB").resize((256, 256))
        mask = Image.open(self.mask_paths[idx]).convert("L").resize((256, 256))
        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

# UNET MODEL DEFINITION
class UNet_KerasStyle(nn.Module):
    def __init__(self):
        super(UNet_KerasStyle, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                #nn.BatchNorm2d(out_ch),  # 讓模型更穩定
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),  # 加入 Dropout，防止過擬合
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                #nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            )

        # 編碼（Contracting Path）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)
        self.conv5 = conv_block(256, 512)

        # 解碼（Expanding Path）

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = conv_block(512, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = conv_block(256, 128)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = conv_block(128, 64)

        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = conv_block(64, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4) # bottleneck

        # 解碼過程 + skip connections
        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        out = torch.sigmoid(self.final(c9))
        return out

def train(model, train_loader, val_loader, device, epochs=10, lr=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    # 在 training loop 外部，建立 list
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'val_precision': [],
        'val_recall': [],
        'val_accuracy': [],
    }


    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # 更新 tqdm 顯示 loss
            loop.set_postfix(loss=train_loss / len(train_loader))

        val_metrics = evaluate_model(model, val_loader, device, criterion)
        # 存入 history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_metrics['loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss / len(train_loader):.4f} - "
              f"Val Loss: {val_metrics['loss']:.4f} - "
              f"IoU: {val_metrics['iou']:.4f} - "
              f"Dice: {val_metrics['dice']:.4f}")
    return history 

# ---------------------
# Metrics
# ---------------------
def iou_pytorch(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum(dim=[1,2,3])
    union = y_pred.sum(dim=[1,2,3]) + y_true.sum(dim=[1,2,3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def dice_coef_pytorch(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum(dim=[1,2,3])
    dice = (2 * intersection + smooth) / (y_pred.sum(dim=[1,2,3]) + y_true.sum(dim=[1,2,3]) + smooth)
    return dice.mean().item()

def precision_pytorch(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    tp = (y_pred * y_true).sum(dim=[1,2,3])
    pp = y_pred.sum(dim=[1,2,3])
    precision = (tp + smooth) / (pp + smooth)
    return precision.mean().item()

def recall_pytorch(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    tp = (y_pred * y_true).sum(dim=[1,2,3])
    possible_positives = y_true.sum(dim=[1,2,3])
    recall = (tp + smooth) / (possible_positives + smooth)
    return recall.mean().item()

def accuracy_pytorch(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    correct = (y_pred == y_true).float()
    return correct.mean().item()

# ---------------------
# Evaluation
# ---------------------
def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0
    iou_sum = 0
    dice_sum = 0
    precision_sum = 0
    recall_sum = 0
    accuracy_sum = 0
    count = 0
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            
            iou_sum += iou_pytorch(outputs, masks) * batch_size
            dice_sum += dice_coef_pytorch(outputs, masks) * batch_size
            precision_sum += precision_pytorch(outputs, masks) * batch_size
            recall_sum += recall_pytorch(outputs, masks) * batch_size
            accuracy_sum += accuracy_pytorch(outputs, masks) * batch_size
            
            count += batch_size
    
    return {
        'loss': running_loss / count,
        'iou': iou_sum / count,
        'dice': dice_sum / count,
        'precision': precision_sum / count,
        'recall': recall_sum / count,
        'accuracy': accuracy_sum / count
    }

def visualize_prediction(model, dataloader, device, num_samples=10):
    model.eval()
    count = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            outputs = (outputs > 0.5).float()

            imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()  # B,H,W,C
            masks = masks.cpu().numpy()  # B,1,H,W
            outputs = outputs.cpu().numpy()  # B,1,H,W

            batch_size = imgs.shape[0]
            for i in range(batch_size):
                if count >= num_samples:
                    return
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(imgs[i])
                axs[0].set_title("Original Image")
                axs[0].axis('off')

                axs[1].imshow(masks[i][0], cmap='gray')
                axs[1].set_title("Ground Truth Mask")
                axs[1].axis('off')

                axs[2].imshow(outputs[i][0], cmap='gray')
                axs[2].set_title("Predicted Mask")
                axs[2].axis('off')

                plt.suptitle(f"Sample {count+1}")
                plt.savefig(f"./predict/prediction_{count+1}.png")
                plt.show()
                plt.close()
                count += 1


# PREPROCESSING FUNCTION
if __name__ == "__main__":
    # 整理檔名順序
    numbers = re.compile(r'(\d+)')
    def numerical_sort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    x_paths = sorted(glob.glob("../skin-lesion/augmented_trainx/*.bmp"), key=numerical_sort)
    y_paths = sorted(glob.glob("../skin-lesion/augmented_trainy/*.bmp"), key=numerical_sort)
        # 切分 train/val/test
    x_trainval, x_test, y_trainval, y_test = train_test_split(x_paths, y_paths, test_size=0.25, random_state=101) # 25% test set
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.5, random_state=101) # 60 % train, 15%
        # 用原圖效果比較好
    transform_img = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_mask = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_KerasStyle().to(device)
    
    mode = int(input("Mode # 0: inference, 1: training: "))
   
    if mode == 0:
        model_path = "unet_epoch50_bs8_lr5e-04.pth"  # 你的模型權重檔
        #state_dict = torch.load("unet_epoch50_bs8_lr5e-04.pth", map_location=device)
        state_dict = torch.load("b_best_50_8_5e-4_05.pth", map_location=device)
        model.load_state_dict(state_dict)
        # 計算所有層的參數總數
        print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("All state_dict params:", sum(p.numel() for p in model.state_dict().values()))
        transform_img = transforms.Compose([transforms.ToTensor()])
        transform_mask = transforms.Compose([transforms.ToTensor()])
        test_dataset  = LesionDataset(x_test, y_test, transform_img, transform_mask, augment=False)
        test_loader  = DataLoader(test_dataset, batch_size=1)

        # 用 visualize_prediction 來可視化預測結果
        visualize_prediction(model, test_loader, device, num_samples=1)
        
    elif mode == 1:
        # 6️⃣ Dataset & DataLoader
        train_dataset = LesionDataset(x_train, y_train, transform_img, transform_mask, augment=True)
        val_dataset   = LesionDataset(x_val, y_val, transform_img, transform_mask, augment=False)
        test_dataset  = LesionDataset(x_test, y_test, transform_img, transform_mask, augment=False)
        # config 
        epochs = 50
        batch_size = 8
        learning_rate = 5e-4
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=8)
        test_loader  = DataLoader(test_dataset, batch_size=1)
        history = train(model, train_loader, val_loader, device, epochs=epochs, lr=learning_rate)
        plot_all_metrics(history, save_dir='./')
        criterion = nn.BCELoss()

        # 儲存模型
        model_filename = f"unet_epoch{epochs}_bs{batch_size}_lr{learning_rate:.0e}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")
        
        for split_name, loader in zip(["Train", "Validation", "Test"], [train_loader, val_loader, test_loader]):
            metrics = evaluate_model(model, loader, device, criterion)
            print(f"\n------- {split_name} Set Metrics -------")
            print(f"Loss:      {metrics['loss']:.4f}")
            print(f"IOU:       {metrics['iou']*100:.2f}%")
            print(f"Dice:      {metrics['dice']*100:.2f}%")
            print(f"Precision: {metrics['precision']*100:.2f}%")
            print(f"Recall:    {metrics['recall']*100:.2f}%")
            print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
            
        # 顯示預測結果對比圖
        visualize_prediction(model, test_loader, device, num_samples=50)