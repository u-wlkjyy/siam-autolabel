
import os
import time
import uuid
import torch
import numpy as np
import random
import math
import sys
from typing import List, Tuple
from siamese_network import SiameseNetwork, get_transforms, predict_similarity
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


class GeetestSolver:
    """极验验证码自动识别解决器"""
    def __init__(self, model_path: str = "./best_siamese_model.pth", headless: bool = False):
        self.model_path = model_path
        self.output_path = "./detect"
        os.makedirs(self.output_path, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.model = self.load_model()
        _, self.transform = get_transforms()

        print("🚀 正在启动浏览器...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)

        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        ]
        self.context = self.browser.new_context(user_agent=random.choice(user_agents))
        self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        self.page = self.context.new_page()

    def load_model(self):
        """加载训练好的孪生神经网络模型"""
        try:
            model = SiameseNetwork(feature_dim=512).to(self.device)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            print("✅ 模型加载成功")
            return model
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

    def human_like_true_delay(self, mean: float = 1.0, std_dev: float = 0.3):
        """模拟更真实的人类延迟（正态分布）"""
        delay = max(0.4, np.random.normal(mean, std_dev))
        time.sleep(delay)

    def simulate_mouse_movement(self, element_locator):
        """模拟人类鼠标移动轨迹，包含悬停和微调"""
        try:
            box = self.page.locator(element_locator).bounding_box()
            if not box:
                return False

            target_x = box['x'] + box['width'] / 2 + random.uniform(-box['width'] / 4, box['width'] / 4)
            target_y = box['y'] + box['height'] / 2 + random.uniform(-box['height'] / 4, box['height'] / 4)

            current_pos = self.page.evaluate('() => [window.mouseX || 0, window.mouseY || 0]')
            start_x, start_y = current_pos if isinstance(current_pos, list) else [0, 0]

            points = []
            steps = random.randint(3, 6)
            for i in range(1, steps):
                progress = i / steps
                offset_x = random.uniform(-30, 30) * math.sin(progress * math.pi)
                offset_y = random.uniform(-20, 20) * math.sin(progress * math.pi * 2)
                points.append((start_x + (target_x - start_x) * progress + offset_x,
                               start_y + (target_y - start_y) * progress + offset_y))

            duration = random.uniform(0.7, 1.4)
            for i, (x, y) in enumerate(points):
                self.page.mouse.move(x, y)
                time.sleep(duration / len(points) * random.uniform(0.8, 1.2))

            self.page.mouse.move(target_x, target_y)

            for _ in range(random.randint(2, 4)):
                self.page.mouse.move(target_x + random.uniform(-4, 4), target_y + random.uniform(-4, 4))
                time.sleep(random.uniform(0.05, 0.15))

            self.page.evaluate(f'() => {{ window.mouseX = {target_x}; window.mouseY = {target_y}; }}')
            return True
        except Exception as e:
            print(f"⚠️ 鼠标移动模拟失败: {e}")
            return False

    def human_like_click(self, element_locator):
        """模拟人类点击行为"""
        if not self.simulate_mouse_movement(element_locator):
            return False
        time.sleep(random.uniform(0.2, 0.6))
        self.page.mouse.down()
        time.sleep(random.uniform(0.05, 0.15))
        self.page.mouse.up()
        time.sleep(random.uniform(0.1, 0.3))
        return True

    def navigate_and_trigger_captcha(self, url: str):
        """导航到页面并触发验证码，返回是否成功触发"""
        try:
            print(f"🌐 正在导航到目标页面...")
            self.page.goto(url, timeout=60000)
            self.human_like_true_delay(2.0, 0.5)

            print("🚶‍♂️ 模拟用户浏览页面...")
            for _ in range(random.randint(1, 2)):
                self.page.mouse.wheel(0, random.randint(200, 500))
                self.human_like_true_delay(1.0, 0.4)

            print("📱 正在填写手机号...")
            self.human_like_click("#phone")
            self.page.keyboard.type("13888888888", delay=random.uniform(50, 200))
            print("✅ 手机号输入完成")

            self.human_like_true_delay(1.0, 0.3)

            button_selector = 'button:has-text("获取验证码")'
            print("🎯 正在点击按钮以触发验证码...")
            if self.human_like_click(button_selector):
                print("✅ 已点击触发按钮")
                return True
            else:
                print("❌ 模拟点击失败，尝试直接点击")
                self.page.locator(button_selector).click()
                return True
        except Exception as e:
            print(f"❌ 导航或触发验证码时出错: {e}")
            return False

    def capture_captcha_images(self) -> str or None:
        """截取验证码相关图片"""
        real_path = os.path.join(self.output_path, uuid.uuid4().hex)
        os.makedirs(real_path, exist_ok=True)
        print(f"📸 正在截取验证码图片到: {os.path.basename(real_path)}")
        try:
            self.page.locator('.geetest_ques_back').screenshot(path=os.path.join(real_path, 'question.png'))
            for i in range(9):
                self.page.locator(f'.geetest_{i}').screenshot(path=os.path.join(real_path, f'geetest_{i}.png'))
            print("✅ 验证码图片截取完毕")
            return real_path
        except Exception as e:
            print(f"❌ 截取验证码图片失败: {e}")
            return None

    def analyze_similarity(self, images_dir: str) -> List[Tuple[int, float]]:
        """分析图片相似度"""
        question_path = os.path.join(images_dir, 'question.png')
        results = []
        print("🔍 正在分析图片相似度...")
        for i in range(9):
            candidate_path = os.path.join(images_dir, f'geetest_{i}.png')
            result = predict_similarity(self.model, question_path, candidate_path, self.transform, self.device)
            results.append((i, result['similarity_score']))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def click_selected_images(self, selected_indices: List[int]):
        """根据索引点击图片"""
        print(f"🎯 准备点击图片: {selected_indices}")
        print("👀 模拟分析验证码...")
        self.human_like_true_delay(1.8, 0.5)

        random.shuffle(selected_indices)
        for i, idx in enumerate(selected_indices):
            print(f"🖱️  正在点击第 {i + 1} 张图片: geetest_{idx}")
            if i > 0: self.human_like_true_delay(0.7, 0.2)
            self.human_like_click(f'.geetest_{idx}')

        print("⏳ 等待验证码自动提交...")
        self.human_like_true_delay(2.0, 0.5)

    def solve(self, similarity_threshold: float = 0.6, max_selections: int = 3) -> bool:
        """完整的验证码解决流程"""
        images_dir = self.capture_captcha_images()
        if not images_dir: return False

        similarity_results = self.analyze_similarity(images_dir)
        print("\n--- 相似度分析结果 ---")
        selected_indices = []
        for idx, score in similarity_results:
            status = "✅ 选中" if score >= similarity_threshold else "❌ 忽略"
            print(f"  图片 {idx}: {score:.4f} -> {status}")
            if score >= similarity_threshold and len(selected_indices) < max_selections:
                selected_indices.append(idx)
        print("----------------------")

        if not selected_indices:
            print("⚠️ 未找到相似度达标的图片，尝试选择最相似的一张")
            selected_indices = [similarity_results[0][0]]

        self.click_selected_images(selected_indices)
        return True

    def close(self):
        """关闭浏览器和清理资源"""
        try:
            self.browser.close()
            self.playwright.stop()
            print("🔒 浏览器已关闭")
        except Exception as e:
            print(f"⚠️ 关闭浏览器时出错: {e}")


def main():
    """主函数，执行全自动在线测试"""
    print("=" * 60)
    print("🎯 极验验证码全自动识别脚本 (在线专用版)")
    print("=" * 60)

    # 检查模型文件是否存在
    model_path = "./best_siamese_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保已完成模型训练并将模型文件放在项目根目录。")
        return

    solver = GeetestSolver(model_path=model_path, headless=False)

    try:
        url = "https://account.siliconflow.cn/zh/login?redirect=https%3A%2F%2Fcloud.siliconflow.cn%2Fsft-d1ab49al401s73aolklg%2Fmodels%3F"
        if not solver.navigate_and_trigger_captcha(url):
            raise RuntimeError("无法触发验证码，程序终止。")

        print("⏳ 正在等待验证码面板加载...")
        # solver.page.wait_for_selector('.geetest_wind', state='visible', timeout=10000)
        print("✅ 验证码面板已出现！")

        # 短暂延迟，确保动画效果完成
        time.sleep(1)

        if solver.solve(similarity_threshold=0.6, max_selections=3):
            print("\n🎉 验证码处理流程完毕！请在浏览器中查看最终结果。")
        else:
            print("\n❌ 验证码处理失败。")

        print("ℹ️  浏览器将在10秒后自动关闭。")
        time.sleep(10)

    except PlaywrightTimeoutError:
        print("\n❌ 等待验证码超时，可能未成功触发或网络问题。")
    except Exception as e:
        print(f"\n❌ 程序执行过程中发生意外错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        solver.close()


if __name__ == "__main__":
    main()
