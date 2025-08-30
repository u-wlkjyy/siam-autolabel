#!/usr/bin/env python3
"""
训练孪生神经网络的脚本
"""

import torch
import os
import argparse
from siamese_network import (
    SiameseNetwork, SiameseDataset, get_transforms,
    train_model, evaluate_model
)
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='训练孪生神经网络')
    parser.add_argument('--dataset_path', type=str, 
                       default='./dataset',
                       help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--feature_dim', type=int, default=512, help='特征维度')
    parser.add_argument('--model_save_path', type=str, default='best_siamese_model.pth',
                       help='模型保存路径')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 获取数据变换
    train_transform, test_transform = get_transforms()
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = SiameseDataset(args.dataset_path, transform=train_transform, mode='train')
    val_dataset = SiameseDataset(args.dataset_path, transform=test_transform, mode='val')
    test_dataset = SiameseDataset(args.dataset_path, transform=test_transform, mode='test')
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # 创建模型
    print(f"创建孪生网络模型（特征维度: {args.feature_dim}）...")
    model = SiameseNetwork(feature_dim=args.feature_dim).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 开始训练
    print("开始训练...")
    train_losses, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device
    )
    
    # 最终测试
    print("进行最终测试...")
    if os.path.exists(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
        test_accuracy = evaluate_model(model, test_loader, device)
        print(f"最终测试准确率: {test_accuracy:.4f}")
    else:
        print("警告: 未找到保存的最佳模型文件")


if __name__ == "__main__":
    main()
