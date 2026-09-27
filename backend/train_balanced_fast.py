import os, sys, torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import cv2

sys.path.insert(0, 'src')
from dataset import get_transforms
from models import get_model

device = torch.device('cpu')
print('Distilling CNN on balanced dataset (136 per class = 408 samples)...')

# Load Teacher (ResNet50)
teacher = get_model('resnet50', num_classes=3).to(device)
t_path = 'models/best_resnet50_final.pth'
teacher.load_state_dict(torch.load(t_path, map_location=device, weights_only=True))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# Load Student (CNN)
student = get_model('cnn', num_classes=3).to(device)
s_path = 'models/best_cnn.pth'
if os.path.exists(s_path):
    student.load_state_dict(torch.load(s_path, map_location=device, weights_only=True))

_, val_transform = get_transforms(img_size=224)
df = pd.read_csv('data/train_balanced.csv')

optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
criterion_ce = nn.CrossEntropyLoss()
criterion_kl = nn.KLDivLoss(reduction='batchmean')

print('Starting fast fine-tuning for 5 epochs on CPU...')
for epoch in range(1, 6):
    student.train()
    total_loss = 0.0
    correct = 0
    total = 0
    df_epoch = df.sample(frac=1.0, random_state=42 + epoch).reset_index(drop=True)
    batch_size = 16
    for b in range(0, len(df_epoch), batch_size):
        chunk = df_epoch.iloc[b:b+batch_size]
        imgs = []
        tabs = []
        targets = []
        for _, row in chunk.iterrows():
            img_path = os.path.join('data/images', row['File name'])
            if not os.path.exists(img_path):
                continue
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_clahe = clahe.apply(img_gray)
            img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor_img = val_transform(pil_img)
            imgs.append(tensor_img)
            tabs.append([
                row['Age'] / 100.0,
                row['Gender'] / 2.0,
                row['Number of missing teeth'] / 32.0,
                float(row['Implant']),
                float(row['Residual root']),
                row['Functional tooth logarithm'] / 32.0
            ])
            targets.append(int(row['Level']) - 1)
        
        if not imgs:
            continue
        x_img = torch.stack(imgs).to(device)
        x_tab = torch.tensor(tabs, dtype=torch.float32).to(device)
        y = torch.tensor(targets, dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        with torch.no_grad():
            t_logits = teacher(x_img, x_tab)
            t_probs = torch.softmax(t_logits / 2.0, dim=1)
        
        s_logits = student(x_img, x_tab)
        s_log_probs = torch.log_softmax(s_logits / 2.0, dim=1)
        
        loss_ce = criterion_ce(s_logits, y)
        loss_kd = criterion_kl(s_log_probs, t_probs) * 4.0
        loss = 0.5 * loss_ce + 0.5 * loss_kd
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(targets)
        preds = torch.argmax(s_logits, dim=1)
        correct += (preds == y).sum().item()
        total += len(targets)
        
    acc = correct / total if total > 0 else 0
    print(f'Epoch {epoch}/5 - Loss: {total_loss/total:.4f} - Train Acc: {acc*100:.1f}%')

torch.save(student.state_dict(), 'models/best_cnn_final.pth')
print('Successfully saved models/best_cnn_final.pth!')
