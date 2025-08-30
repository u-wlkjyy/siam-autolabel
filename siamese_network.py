import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
import numpy as np
import random

class SiameseDataset(Dataset):
    """孪生网络数据集类"""
    
    def __init__(self, dataset_path: str, transform=None, mode='train', train_ratio=0.8):
        """
        Args:
            dataset_path: 数据集根目录路径
            transform: 图像变换
            mode: 'train' 或 'test' 或 'val'
            train_ratio: 训练集比例
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.mode = mode
        
        # 获取所有文件夹
        self.folders = [f for f in os.listdir(dataset_path) 
                       if os.path.isdir(os.path.join(dataset_path, f))]
        
        # 按比例分割数据集
        random.shuffle(self.folders)
        split_idx = int(len(self.folders) * train_ratio)
        
        if mode == 'train':
            self.folders = self.folders[:split_idx]
        elif mode == 'test':
            self.folders = self.folders[split_idx:]
        elif mode == 'val':
            # 使用训练集的一部分作为验证集
            val_split = int(split_idx * 0.8)
            self.folders = self.folders[val_split:split_idx]
        
        print(f"{mode.upper()}集包含 {len(self.folders)} 个样本文件夹")
        
    def __len__(self):
        # 每个文件夹生成多个正负样本对
        return len(self.folders) * 20  # 每个文件夹生成20个样本对
    
    def __getitem__(self, idx):
        """生成一个样本对和标签"""
        folder_idx = idx // 20
        folder_name = self.folders[folder_idx]
        folder_path = os.path.join(self.dataset_path, folder_name)
        
        # 读取问题图片
        question_path = os.path.join(folder_path, 'question.png')
        if not os.path.exists(question_path):
            # 如果没有question.png，随机选择一个图片作为anchor
            all_images = glob.glob(os.path.join(folder_path, '*.png'))
            question_path = random.choice(all_images)
        
        question_img = Image.open(question_path).convert('RGB')
        
        # 获取正样本（答案）和负样本
        answer_files = glob.glob(os.path.join(folder_path, 'geetest_answer_*.png'))
        negative_files = glob.glob(os.path.join(folder_path, 'geetest_[0-9].png'))
        
        # 随机决定生成正样本对还是负样本对
        is_positive = random.random() > 0.5
        
        if is_positive and answer_files:
            # 生成正样本对
            pair_path = random.choice(answer_files)
            label = 1
        elif negative_files:
            # 生成负样本对
            pair_path = random.choice(negative_files)
            label = 0
        else:
            # 如果没有足够的样本，随机选择
            all_images = glob.glob(os.path.join(folder_path, '*.png'))
            all_images = [img for img in all_images if img != question_path]
            if all_images:
                pair_path = random.choice(all_images)
                label = 1 if 'answer' in pair_path else 0
            else:
                # 最后的fallback
                pair_path = question_path
                label = 1
        
        pair_img = Image.open(pair_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            question_img = self.transform(question_img)
            pair_img = self.transform(pair_img)
        
        return question_img, pair_img, torch.tensor(label, dtype=torch.float32)


class ResNetBackbone(nn.Module):
    """ResNet特征提取器"""
    
    def __init__(self, pretrained=True, feature_dim=512):
        super(ResNetBackbone, self).__init__()
        
        # 使用预训练的ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # 移除最后的分类层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 添加特征降维层
        self.feature_projector = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x):
        # 提取特征
        features = self.resnet(x)
        features = features.view(features.size(0), -1)  # 展平
        
        # 特征投影
        features = self.feature_projector(features)
        
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        
        return features


class SiameseNetwork(nn.Module):
    """孪生神经网络"""
    
    def __init__(self, feature_dim=512):
        super(SiameseNetwork, self).__init__()
        
        # 共享的特征提取器
        self.backbone = ResNetBackbone(pretrained=True, feature_dim=feature_dim)
        
        # 相似度计算层
        self.similarity_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img1, img2):
        # 提取两个图像的特征
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        
        # 计算特征距离（可选方法）
        euclidean_distance = F.pairwise_distance(feat1, feat2)
        cosine_similarity = F.cosine_similarity(feat1, feat2)
        
        # 连接特征进行相似度预测
        combined_features = torch.cat([feat1, feat2], dim=1)
        similarity_score = self.similarity_head(combined_features)
        
        return similarity_score.squeeze(), euclidean_distance, cosine_similarity


class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, distance, label):
        # 对比损失计算
        loss = torch.mean(
            label * torch.pow(distance, 2) + 
            (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        return loss


class TripletLoss(nn.Module):
    """三元组损失函数"""
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        loss = torch.mean(torch.clamp(
            distance_positive - distance_negative + self.margin, min=0.0))
        
        return loss


def get_transforms():
    """获取数据变换"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """训练模型"""
    import time
    import sys
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # 多种损失函数
    bce_loss = nn.BCELoss()
    contrastive_loss = ContrastiveLoss(margin=1.0)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            similarity_scores, euclidean_dist, cosine_sim = model(img1, img2)
            
            # 计算损失（结合多种损失）
            bce_loss_val = bce_loss(similarity_scores, labels)
            contrastive_loss_val = contrastive_loss(euclidean_dist, labels)
            
            # 总损失
            total_loss_val = bce_loss_val + 0.5 * contrastive_loss_val
            
            # 反向传播
            total_loss_val.backward()
            optimizer.step()
            
            # 更新损失统计
            total_loss += total_loss_val.item()
            running_loss += total_loss_val.item()
            
            # 计算进度
            progress = (batch_idx + 1) / len(train_loader)
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '▒' * (bar_length - filled_length)
            
            # 计算速度和ETA
            elapsed_time = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed_time if elapsed_time > 0 else 0
            eta_seconds = (len(train_loader) - batch_idx - 1) / batches_per_sec if batches_per_sec > 0 else 0
            eta_str = f"{int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"

            # 显示进度
            avg_loss = running_loss / (batch_idx + 1)
            sys.stdout.write(f'\rEpoch {epoch+1}/{num_epochs} |{bar}| '
                           f'{batch_idx+1}/{len(train_loader)} '
                           f'[{progress*100:5.1f}%] '
                           f'loss: {avg_loss:.4f} '
                           f'ETA: {eta_str} '
                           f'{batches_per_sec:.1f}it/s')
            sys.stdout.flush()
        
        # 换行并显示epoch总结
        print()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        print("Validating...")
        val_acc = evaluate_model(model, val_loader, device, verbose=False)  # 验证时不显示详细信息
        val_accuracies.append(val_acc)
        
        # epoch总结
        epoch_time = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'train_loss={avg_train_loss:.4f} '
              f'val_acc={val_acc:.4f} '
              f'lr={lr:.6f} '
              f'time={epoch_time:.1f}s')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_siamese_model.pth')
            print(f'🏆 New best model saved! Val Acc: {best_val_acc:.4f}')
        
        scheduler.step()
        print("-" * 80)
    
    return train_losses, val_accuracies


def evaluate_model(model, test_loader, device='cuda', verbose=True):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            similarity_scores, _, _ = model(img1, img2)
            predictions = (similarity_scores > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算基本准确率
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_predictions == all_labels)
    
    if verbose:
            print(f'准确率: {accuracy:.4f}')
            # 手动计算精确率和召回率
            tp = np.sum((all_predictions == 1) & (all_labels == 1))
            fp = np.sum((all_predictions == 1) & (all_labels == 0))
            fn = np.sum((all_predictions == 0) & (all_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            print(f'精确率: {precision:.4f}')
            print(f'召回率: {recall:.4f}')
            print(f'F1分数: {f1:.4f}')
    
    return accuracy


def predict_similarity(model, img1_path, img2_path, transform, device='cuda'):
    """预测两张图片的相似度"""
    model.eval()
    
    # 加载图片
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # 应用变换
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        similarity_score, euclidean_dist, cosine_sim = model(img1, img2)
    
    return {
        'similarity_score': similarity_score.item(),
        'euclidean_distance': euclidean_dist.item(),
        'cosine_similarity': cosine_sim.item(),
        'is_similar': similarity_score.item() > 0.5
    }