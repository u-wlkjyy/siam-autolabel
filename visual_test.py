#!/usr/bin/env python3

import torch
import os
import glob
import cv2
import numpy as np
from PIL import Image
from siamese_network import SiameseNetwork, get_transforms, predict_similarity


class SimpleVisualTester:
    
    def __init__(self, model_path, test_dir, feature_dim=512):
        self.model_path = model_path
        self.test_dir = test_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        
        # 加载模型
        print(f"Model path: {model_path}")
        print(f"Device: {self.device}")
        
        self.model = SiameseNetwork(feature_dim=feature_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 获取数据变换
        _, self.transform = get_transforms()
        
        # 获取所有测试目录
        self.test_folders = self._get_test_folders()
        self.current_index = 0
        
        print(f"📁 找到 {len(self.test_folders)} 个测试目录")
        
        # 设置显示参数
        self.img_width = 200
        self.img_height = 150
        self.cols = 4
        
    def _get_test_folders(self):
        """获取所有测试目录"""
        folders = []
        if os.path.exists(self.test_dir):
            for item in os.listdir(self.test_dir):
                folder_path = os.path.join(self.test_dir, item)
                if os.path.isdir(folder_path):
                    # 检查是否包含question.png和geetest_*.png文件
                    question_file = os.path.join(folder_path, 'question.png')
                    geetest_files = glob.glob(os.path.join(folder_path, 'geetest_*.png'))
                    
                    if os.path.exists(question_file) and geetest_files:
                        folders.append(folder_path)
        
        return sorted(folders)
    
    def _load_and_resize_image(self, image_path):
        """加载并调整图片大小"""
        try:
            # 使用PIL读取图片然后转换为OpenCV格式
            pil_img = Image.open(image_path).convert('RGB')
            pil_img = pil_img.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)
            
            # 转换为OpenCV格式 (BGR)
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            return cv_img
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            # 返回黑色图片
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
    
    def _add_text_to_image(self, img, text, position, color=(255, 255, 255), font_scale=0.5, thickness=1):
        """在图片上添加文字"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 处理多行文本
        lines = text.split('\n')
        y_offset = 0
        
        for line in lines:
            if line.strip():  # 跳过空行
                # 计算文字大小来调整位置
                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                
                # 添加文字背景
                cv2.rectangle(img, 
                             (position[0] - 2, position[1] + y_offset - text_height - 2),
                             (position[0] + text_width + 2, position[1] + y_offset + baseline + 2),
                             (0, 0, 0), -1)
                
                # 添加文字
                cv2.putText(img, line, (position[0], position[1] + y_offset), 
                           font, font_scale, color, thickness, cv2.LINE_AA)
                
                y_offset += text_height + 5
    
    def _create_visualization(self, folder_path):
        """创建当前目录的可视化结果"""
        question_path = os.path.join(folder_path, 'question.png')
        geetest_files = sorted(glob.glob(os.path.join(folder_path, 'geetest_*.png')))
        
        if not geetest_files:
            return None, []
        
        print(f"\n📂 处理目录: {os.path.basename(folder_path)}")
        print("🔄 正在预测相似度...")
        
        # 计算所有相似度
        results = []
        for i, geetest_file in enumerate(geetest_files):
            try:
                result = predict_similarity(
                    self.model, question_path, geetest_file, 
                    self.transform, self.device
                )
                results.append({
                    'file': geetest_file,
                    'filename': os.path.basename(geetest_file),
                    'similarity_score': result['similarity_score'],
                    'euclidean_distance': result['euclidean_distance'],
                    'cosine_similarity': result['cosine_similarity'],
                    'is_similar': result['is_similar']
                })
                print(f"  ✓ {os.path.basename(geetest_file)}: {result['similarity_score']:.3f}")
            except Exception as e:
                print(f"  ❌ 预测失败 {geetest_file}: {e}")
                continue
        
        if not results:
            return None, []
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # 创建组合图像
        question_img = self._load_and_resize_image(question_path)
        
        # 计算布局
        total_images = len(results) + 1  # +1 for question image
        rows = (total_images + self.cols - 1) // self.cols
        
        # 创建画布
        canvas_width = self.cols * self.img_width
        canvas_height = rows * (self.img_height + 40)  # +40 for text
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 添加问题图片
        y_pos = 0
        x_pos = 0
        canvas[y_pos:y_pos+self.img_height, x_pos:x_pos+self.img_width] = question_img
        
        # 在问题图片上添加标题
        self._add_text_to_image(canvas, "QUESTION", (x_pos + 5, y_pos + 20), 
                               (0, 255, 255), 0.7, 2)  # 黄色
        
        # 添加候选图片
        for i, result in enumerate(results):
            row = (i + 1) // self.cols
            col = (i + 1) % self.cols
            
            y_pos = row * (self.img_height + 40)
            x_pos = col * self.img_width
            
            # 加载图片
            img = self._load_and_resize_image(result['file'])
            
            # 取置信度前三
            if i <= 2:
                border_color = (0, 255, 0)  # 绿色 - 相似
                status_text = "SIMILAR"
                status_color = (0, 255, 0)
            else:
                border_color = (0, 0, 255)  # 红色 - 不相似
                status_text = "DIFFERENT"
                status_color = (0, 0, 255)
            
            # 添加边框
            cv2.rectangle(img, (0, 0), (self.img_width-1, self.img_height-1), border_color, 3)
            
            # 将图片放到画布上
            canvas[y_pos:y_pos+self.img_height, x_pos:x_pos+self.img_width] = img
            
            # 添加文字信息
            info_text = f"{result['filename']}\n{status_text}\nScore: {result['similarity_score']:.3f}"
            self._add_text_to_image(canvas, info_text, 
                                  (x_pos + 5, y_pos + self.img_height + 15), 
                                  status_color, 0.4, 1)
        
        # 添加标题
        title = f"Test: {os.path.basename(folder_path)}"
        self._add_text_to_image(canvas, title, (10, 30), (255, 255, 255), 0.8, 2)
        
        return canvas, results
    
    def run_interactive_test(self):
        """运行交互式测试"""
        print("\n" + "="*80)
        print("🎮 交互式可视化测试")
        print("📖 操作说明:")
        print("   空格键 - 下一个测试目录")
        print("   'q'键  - 退出程序")
        print("   'p'键  - 上一个测试目录") 
        print("   's'键  - 保存当前结果图片")
        print("="*80)
        
        cv2.namedWindow('Siamese Network Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Siamese Network Test', 1200, 800)
        
        while self.current_index < len(self.test_folders):
            current_folder = self.test_folders[self.current_index]
            folder_name = os.path.basename(current_folder)
            
            print(f"\n📂 [{self.current_index + 1}/{len(self.test_folders)}] 测试: {folder_name}")
            
            # 创建可视化
            try:
                canvas, results = self._create_visualization(current_folder)
                
                if canvas is None:
                    print("❌ 无法处理当前目录，跳过...")
                    self.current_index += 1
                    continue
                
                # 显示结果统计
                similar_count = sum(1 for r in results if r['is_similar'])
                total_count = len(results)
                avg_similarity = np.mean([r['similarity_score'] for r in results])
                best_match = max(results, key=lambda x: x['similarity_score'])
                
                print(f"\n📊 结果统计:")
                print(f"  相似图片: {similar_count}/{total_count}")
                print(f"  平均相似度: {avg_similarity:.3f}")
                print(f"  最佳匹配: {best_match['filename']} (相似度: {best_match['similarity_score']:.3f})")
                
                # 添加统计信息到画布
                stats_text = f"Similar: {similar_count}/{total_count} | Avg: {avg_similarity:.3f} | Best: {best_match['filename']} ({best_match['similarity_score']:.3f})"
                self._add_text_to_image(canvas, stats_text, (10, canvas.shape[0] - 20), 
                                      (255, 255, 0), 0.5, 1)  # 青色
                
                # 显示图片
                cv2.imshow('Siamese Network Test', canvas)
                
                print(f"\n⌨️  按键操作: 空格(下一个) | p(上一个) | s(保存) | q(退出)")
                
                # 等待按键
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    
                    if key == ord(' ') or key == 13:  # 空格或回车
                        self.current_index += 1
                        break
                    elif key == ord('q'):  # 退出
                        cv2.destroyAllWindows()
                        print("👋 退出测试")
                        return
                    elif key == ord('p'):  # 上一个
                        if self.current_index > 0:
                            self.current_index -= 1
                        else:
                            print("⚠️  已经是第一个目录")
                            continue
                        break
                    elif key == ord('s'):  # 保存
                        save_path = f'test_result_{folder_name}.png'
                        cv2.imwrite(save_path, canvas)
                        print(f"💾 结果已保存到: {save_path}")
                    else:
                        print(f"❓ 按键无效: {chr(key) if key < 128 else key}")
                
            except Exception as e:
                print(f"❌ 处理目录时出错: {e}")
                import traceback
                traceback.print_exc()
                self.current_index += 1
                continue
        
        cv2.destroyAllWindows()
        print("\n🎉 所有测试目录已完成!")

def main():
    """主函数"""
    
    # 配置参数
    model_path = './best_siamese_model.pth'
    test_dir = './tests'
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        return
    
    # 创建测试器
    try:
        SimpleVisualTester(model_path, test_dir).run_interactive_test()
    except Exception as e:
        print(f"❌ 初始化测试器失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
