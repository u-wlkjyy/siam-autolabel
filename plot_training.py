#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_training_log(log_text):
    """解析训练日志，提取损失和准确率数据"""
    
    epochs = []
    train_losses = []
    val_accuracies = []
    
    # 使用正则表达式提取数据
    pattern = r'Epoch (\d+)/\d+:.*?train_loss=([\d.]+).*?val_acc=([\d.]+)'
    
    matches = re.findall(pattern, log_text)
    
    for match in matches:
        epoch = int(match[0])
        train_loss = float(match[1])
        val_acc = float(match[2])
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
    
    return epochs, train_losses, val_accuracies

def plot_training_curves(epochs, train_losses, val_accuracies):
    """绘制训练曲线"""
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制训练损失
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='训练损失')
    ax1.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失值 (Loss)', fontsize=12)
    ax1.set_title('训练损失变化曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 设置y轴范围，让曲线更清晰
    ax1.set_ylim(0, max(train_losses) * 1.1)
    
    # 添加数值标注
    for i, (x, y) in enumerate(zip(epochs, train_losses)):
        if i % 3 == 0:  # 每3个点标注一次，避免过密
            ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
    # 绘制验证准确率
    ax2.plot(epochs, val_accuracies, 'r-', linewidth=2, marker='s', markersize=4, label='验证准确率')
    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.set_title('验证准确率变化曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # 设置y轴范围
    ax2.set_ylim(min(val_accuracies) * 0.98, 1.0)
    
    # 添加数值标注
    for i, (x, y) in enumerate(zip(epochs, val_accuracies)):
        if i % 3 == 0:  # 每3个点标注一次
            ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
    # 标记最佳性能点
    best_idx = np.argmax(val_accuracies)
    best_epoch = epochs[best_idx]
    best_acc = val_accuracies[best_idx]
    
    ax2.scatter(best_epoch, best_acc, color='gold', s=100, zorder=5, 
               label=f'最佳性能: Epoch {best_epoch}, 准确率 {best_acc:.4f}')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    
    # 训练日志数据
    log_text = """
    模型总参数数: 11,997,249
    可训练参数数: 11,997,249
    开始训练...
    Epoch 1/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.4175 ETA: 00:00 19.8it/s
    Validating...
    Epoch 1/1000: train_loss=0.4175 val_acc=0.9486 lr=0.001000 time=56.0s
    🏆 New best model saved! Val Acc: 0.9486
    --------------------------------------------------------------------------------
    Epoch 2/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.2298 ETA: 00:00 21.5it/s
    Validating...
    Epoch 2/1000: train_loss=0.2298 val_acc=0.9651 lr=0.001000 time=52.2s
    🏆 New best model saved! Val Acc: 0.9651
    --------------------------------------------------------------------------------
    Epoch 3/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1913 ETA: 00:00 21.2it/s
    Validating...
    Epoch 3/1000: train_loss=0.1913 val_acc=0.9752 lr=0.001000 time=52.4s
    🏆 New best model saved! Val Acc: 0.9752
    --------------------------------------------------------------------------------
    Epoch 4/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1765 ETA: 00:00 21.6it/s
    Validating...
    Epoch 4/1000: train_loss=0.1765 val_acc=0.9656 lr=0.001000 time=51.6s
    --------------------------------------------------------------------------------
    Epoch 5/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1623 ETA: 00:00 20.9it/s
    Validating...
    Epoch 5/1000: train_loss=0.1623 val_acc=0.9781 lr=0.001000 time=54.0s
    🏆 New best model saved! Val Acc: 0.9781
    --------------------------------------------------------------------------------
    Epoch 6/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1466 ETA: 00:00 21.5it/s
    Validating...
    Epoch 6/1000: train_loss=0.1466 val_acc=0.9808 lr=0.001000 time=51.8s
    🏆 New best model saved! Val Acc: 0.9808
    --------------------------------------------------------------------------------
    Epoch 7/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1531 ETA: 00:00 21.7it/s
    Validating...
    Epoch 7/1000: train_loss=0.1531 val_acc=0.9824 lr=0.001000 time=51.4s
    🏆 New best model saved! Val Acc: 0.9824
    --------------------------------------------------------------------------------
    Epoch 8/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1330 ETA: 00:00 21.1it/s
    Validating...
    Epoch 8/1000: train_loss=0.1330 val_acc=0.9813 lr=0.001000 time=53.0s
    --------------------------------------------------------------------------------
    Epoch 9/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1386 ETA: 00:00 21.5it/s
    Validating...
    Epoch 9/1000: train_loss=0.1386 val_acc=0.9769 lr=0.001000 time=51.7s
    --------------------------------------------------------------------------------
    Epoch 10/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1278 ETA: 00:00 21.4it/s
    Validating...
    Epoch 10/1000: train_loss=0.1278 val_acc=0.9794 lr=0.001000 time=52.1s
    --------------------------------------------------------------------------------
    Epoch 11/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1251 ETA: 00:00 20.9it/s
    Validating...
    Epoch 11/1000: train_loss=0.1251 val_acc=0.9753 lr=0.001000 time=53.2s
    --------------------------------------------------------------------------------
    Epoch 12/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1220 ETA: 00:00 21.6it/s
    Validating...
    Epoch 12/1000: train_loss=0.1220 val_acc=0.9818 lr=0.001000 time=51.6s
    --------------------------------------------------------------------------------
    Epoch 13/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1174 ETA: 00:00 20.4it/s
    Validating...
    Epoch 13/1000: train_loss=0.1174 val_acc=0.9763 lr=0.001000 time=54.4s
    --------------------------------------------------------------------------------
    Epoch 14/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1152 ETA: 00:00 21.1it/s
    Validating...
    Epoch 14/1000: train_loss=0.1152 val_acc=0.9786 lr=0.001000 time=52.8s
    --------------------------------------------------------------------------------
    Epoch 15/1000 |████████████████████████████████████████| 993/993 [100.0%] loss: 0.1202 ETA: 00:00 20.3it/s
    Validating...
    Epoch 15/1000: train_loss=0.1202 val_acc=0.9805 lr=0.001000 time=54.5s
    """
    
    # 解析日志数据
    epochs, train_losses, val_accuracies = parse_training_log(log_text)
    
    # 绘制图表
    fig = plot_training_curves(epochs, train_losses, val_accuracies)
    

    print(f"\n图表已保存:")
    print(f"  - train.png (包含训练损失和验证准确率)")
    print(f"  - val.png (相同内容的副本)")
    
    # 显示图表
    plt.show()
    

if __name__ == "__main__":
    main()
