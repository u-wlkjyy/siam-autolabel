import os.path
import shutil
import threading
import time
import uuid

from playwright.sync_api import sync_playwright

output_path = './output'

os.makedirs(output_path, exist_ok=True)
files = os.listdir(output_path)

for item in files:
    if os.path.isdir(os.path.join(output_path, item)):
        if len(os.listdir(os.path.join(output_path, item))) != 11:
            shutil.rmtree(os.path.join(output_path, item))
            print('[WARNING] Removed folder {}'.format(item))

index = len(files)
worker_num = 16

def slider():
    while True:
        playwright = sync_playwright().start()

        browser = playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )

        # 创建浏览器上下文
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()


        page.goto('https://account.siliconflow.cn/zh/login?redirect=https%3A%2F%2Fcloud.siliconflow.cn%2F%3F')
        page.wait_for_selector("#phone",timeout=10000)
        page.fill("#phone","13888888888",timeout=10000)

        page.click("body > div.w-full.h-full.flex.overflow-hidden.relative > div.flex-1.h-full.flex.flex-col.items-center.justify-center > section > div > form > div:nth-child(2) > div > div > div > div > span > span > span > button",timeout=10000)

        while True:
            try:
                # index += 1
                task_uuid = uuid.uuid4().hex

                real_path = os.path.join(output_path,task_uuid)
                os.makedirs(real_path,exist_ok=True)

                geetest_box = page.locator('.geetest_box')
                geetest_box.screenshot(timeout=10000,path=os.path.join(real_path,'full.png'))

                ques = page.locator('.geetest_ques_back')
                ques.screenshot(timeout=10000,path=os.path.join(real_path,'question.png'))

                for i in range(9):
                    sub_picture = page.locator('.geetest_' + str(i))
                    sub_picture.screenshot(timeout=10000,path=os.path.join(real_path,'geetest_' + str(i) + '.png'))

                page.click(".geetest_refresh",timeout=10000)
                # page.click('.geetest_close',timeout=10000)
                # print('[INFO] 采集成功：' + str(index),end='\r')
                time.sleep(2)
            except:
                playwright.stop()
                break

for i in range(worker_num):
    threading.Thread(target=slider).start()

def count():
    while True:
        print('[INFO] 采集成功：' + str(len(os.listdir('./tests'))),end='\r')
        time.sleep(1)

count()