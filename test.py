from siamese_network import SiameseNetwork, get_transforms, predict_similarity
import torch

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork(feature_dim=512).to(device)
model.load_state_dict(torch.load('best_siamese_model.pth', map_location=device))

# 获取数据变换
_, transform = get_transforms()

# 预测相似度
result = predict_similarity(
    model=model,
    img1_path='question.png',
    img2_path='geetest_7.png', # 选项图片
    transform=transform,
    device=device
)

print(f"相似度分数: {result['similarity_score']:.3f}")
print(f"是否相似: {result['is_similar']}")