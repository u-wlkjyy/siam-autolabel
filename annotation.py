#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import requests
import base64
import os
from typing import Optional, Dict, Any, List



class GeminiClient:

    def __init__(self, api_key: str, base_url: str = os.getenv("BASE_URL")):
        """
        初始化客户端

        Args:
            api_key: API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Python-Client/1.0'
        })

    def encode_image(self, image_path: str) -> str:
        """
        将图片编码为base64格式

        Args:
            image_path: 图片文件路径

        Returns:
            base64编码的图片字符串
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string

    def chat_completion(self,
                        messages: List[Dict[str, Any]],
                        model: str = "gemini-2.5-pro",
                        temperature: float = 0.7,
                        max_tokens: Optional[int] = None,
                        stream: bool = False) -> Dict[str, Any]:
        """
        发送聊天完成请求

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数 (0-1)
            max_tokens: 最大token数
            stream: 是否流式响应

        Returns:
            API响应结果
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"API请求失败: {e}")


    def analyze_image(self,
                      image_path: str,
                      prompt: str = "请描述这张图片",
                      model: str = "gemini-2.5-pro") -> str:
        """
        分析图片内容

        Args:
            image_path: 图片文件路径
            prompt: 分析提示词
            model: 使用的模型

        Returns:
            分析结果文本
        """
        try:
            # 编码图片
            base64_image = self.encode_image(image_path)

            # 构造消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""(taskId: {prompt}) 你是一个专业的验证码识别专家，具有卓越的图像识别能力。请仔细分析这张验证码图片：
        
        ## 任务目标
        这是一个图片选择类验证码，需要根据给定的条件选择正确的图片。常见类型包括：
        - 选择包含特定物体的图片（汽车、红绿灯、人行横道等）
        - 选择包含特定动物的图片（鸟、猫、狗等）
        - 选择包含特定场景的图片（商店、房屋、桥梁等）
        
        ## 图片网格布局
        图片采用标准3×3网格布局，位置编号如下：
        ```
        1  2  3
        4  5  6  
        7  8  9
        ```
        
        ## 详细分析流程
        1. **读取指令**: 仔细查看图片中的文字指令（通常在顶部、底部或侧边）
        2. **逐个分析**: 按1-9顺序检查每个网格中的图片内容
        3. **特征识别**: 识别每张图片的主要物体、场景和特征
        4. **条件匹配**: 将每张图片与验证码要求进行精确对比
        5. **结果确认**: 确保选择的图片完全符合条件要求
        
        ## 识别要点
        - 注意图片的细节特征，不要被相似物体误导
        - 考虑图片的角度、光线和部分遮挡情况
        - 如果图片模糊或不确定，采用保守策略
        - 优先选择明确符合条件的图片
        
        ## 严格输出要求
        - 仅输出符合条件的图片编号
        - 多个编号用英文逗号分隔，无空格
        - 不输出任何解释、分析过程或其他文字
        - 输出格式示例：1,5,7 或 2 或 3,6,9
        
        现在请开始分析："""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # 发送请求
            response = self.chat_completion(messages=messages, model=model)

            # 提取回复内容
            if 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['message']['content']
            else:
                raise Exception("响应格式异常，未找到回复内容")

        except Exception as e:
            raise Exception(f"图片分析失败: {e}")


def is_already_annotated(file_name):
    """检查文件是否已经被标注过"""
    dataset_path = os.path.join("./dataset", file_name)
    if not os.path.exists(dataset_path):
        return False
    
    # 检查是否存在答案文件（geetest_answer_*.png）
    files_in_dataset = os.listdir(dataset_path)
    answer_files = [f for f in files_in_dataset if f.startswith("geetest_answer_")]
    
    return len(answer_files) > 0


def process_single_file(client, file_name, print_lock):
    """处理单个文件的标注"""
    with print_lock:
        print(f"[INFO] 线程 {threading.current_thread().name} 开始分析：{file_name}")
    
    # 检查是否已经标注过
    if is_already_annotated(file_name):
        with print_lock:
            print(f"[SKIP] {file_name} 已经标注过，跳过处理")
        return True, file_name
    
    image_path = os.path.join("./output", file_name, "full.png")
    
    if not os.path.exists(image_path):
        with print_lock:
            print(f"[SKIP] 未找到图片文件：{image_path}")
        return False, file_name
    
    # 重试机制
    for retry in range(3):
        try:
            with print_lock:
                print(f"[INFO] {file_name} 第 {retry + 1} 次尝试分析")
            
            response = client.analyze_image(
                image_path=image_path,
                prompt=file_name,
                model="gemini-2.5-pro"
            )
            
            with print_lock:
                print(f"[RESULT] {file_name} 分析结果: {response}")
            
            results = response.split(",")
            
            # 创建数据集目录并复制文件
            desc_path = os.path.join("./dataset", file_name)
            src_path = os.path.join("./output", file_name)
            
            # 确保dataset目录存在
            os.makedirs("./dataset", exist_ok=True)
            
            # 如果目标目录已存在，先删除
            if os.path.exists(desc_path):
                shutil.rmtree(desc_path)
            
            shutil.copytree(src_path, desc_path)
            
            # 删除full.png
            full_png_path = os.path.join(desc_path, "full.png")
            if os.path.exists(full_png_path):
                os.remove(full_png_path)
            
            # 重命名答案文件
            for i in results:
                i = i.strip()  # 去除可能的空格
                if not i.isdigit():
                    continue
                
                old_name = os.path.join(desc_path, f"geetest_{int(i) - 1}.png")
                new_name = os.path.join(desc_path, f"geetest_answer_{int(i) - 1}.png")
                
                if os.path.exists(old_name):
                    os.rename(old_name, new_name)
            
            with print_lock:
                print(f'[SUCCESS] {file_name} 处理完成')
            
            return True, file_name
            
        except Exception as e:
            with print_lock:
                print(f"[ERROR] {file_name} 第 {retry + 1} 次分析失败: {e}")
            
            if retry == 2:  # 最后一次重试失败
                with print_lock:
                    print(f"[FAILED] {file_name} 达到最大重试次数，处理失败")
                return False, file_name
    
    return False, file_name


def main():
    """主函数 - 使用多线程进行标注"""
    
    # 初始化客户端
    api_key = os.getenv("API_KEY")
    client = GeminiClient(api_key)
    
    # 获取所有待处理文件
    if not os.path.exists("./output"):
        print("[ERROR] output 目录不存在")
        return
    
    files = os.listdir("./output")
    files = [f for f in files if os.path.isdir(os.path.join("./output", f))]
    
    if not files:
        print("[INFO] 没有找到需要处理的文件")
        return
    
    print(f"[INFO] 找到 {len(files)} 个文件需要处理")
    
    # 过滤已经标注过的文件
    unprocessed_files = []
    for file_name in files:
        if not is_already_annotated(file_name):
            unprocessed_files.append(file_name)
        else:
            print(f"[SKIP] {file_name} 已经标注过，跳过")
    
    if not unprocessed_files:
        print("[INFO] 所有文件都已经标注过了")
        return
    
    print(f"[INFO] 需要处理 {len(unprocessed_files)} 个未标注的文件")
    
    # 创建线程锁用于打印同步
    print_lock = Lock()
    
    # 使用8个线程进行并发处理
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_file, client, file_name, print_lock): file_name 
            for file_name in unprocessed_files
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                success, processed_file = future.result()
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                with print_lock:
                    print(f"[EXCEPTION] {file_name} 处理过程中发生异常: {e}")
                failed_count += 1
    
    print(f"\n[SUMMARY] 处理完成！")
    print(f"[SUMMARY] 成功: {success_count} 个文件")
    print(f"[SUMMARY] 失败: {failed_count} 个文件")
    print(f"[SUMMARY] 总计: {len(unprocessed_files)} 个文件")


if __name__ == "__main__":
    main()
