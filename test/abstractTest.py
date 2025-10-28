import requests
from bs4 import BeautifulSoup
import os
import json
from typing import Dict, List, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class LLVMAutoSummary:
    def __init__(self, llm_model: str = "deepseek-chat", cache_dir: str = "./llvm_doxygen_summaries",
                 max_workers: int = 8):
        """
        初始化LLVM源码Doxygen文档摘要生成器
        :param max_workers: 最大线程数
        """
        self.llm_model = llm_model
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        os.makedirs(cache_dir, exist_ok=True)

        # 初始化DeepSeek客户端
        self.client = OpenAI(
            api_key="sk-ad1a7b32b3f2419db17ed342a23b6b06",
            base_url="https://api.deepseek.com/v1"
        )

        # 线程锁：避免多线程同时打印导致输出混乱
        self.print_lock = threading.Lock()

    def crawl_llvm_doxygen_links(self, base_url: str = "https://llvm.org/doxygen/") -> Dict[str, str]:
        """爬取LLVM源码的Doxygen文档链接（串行，仅执行一次）"""
        file_url_map = {}
        try:
            index_url = f"{base_url}files.html"
            response = requests.get(index_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a", href=True):
                href = link["href"]
                if any(ext in href for ext in ["_8cpp.html", "_8h.html", "_8c.html"]) and not href.startswith("http"):
                    raw_file_name = href.replace("_8cpp.html", ".cpp") \
                        .replace("_8h.html", ".h") \
                        .replace("_8c.html", ".c")
                    full_url = f"{base_url.rstrip('/')}/{href}"
                    file_url_map[raw_file_name] = full_url

            with self.print_lock:
                print(f"成功爬取 {len(file_url_map)} 个LLVM源码的Doxygen文档链接")
        except Exception as e:
            with self.print_lock:
                print(f"Doxygen文档爬取失败：{str(e)}")
        return file_url_map

    def fetch_doxygen_content(self, url: str) -> str:
        """提取Doxygen文档核心注释内容"""
        try:
            response = requests.get(url, timeout=15)  # 多线程下适当延长超时时间
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content_blocks = soup.find_all(["div", "p"], class_=["contents", "textblock"])
            if not content_blocks:
                content_blocks = soup.find_all(id="details")
            raw_content = "\n".join(
                [block.get_text(strip=True) for block in content_blocks if block.get_text(strip=True)]
            )
            return raw_content[:8000]
        except Exception as e:
            with self.print_lock:
                print(f"[内容提取失败] URL: {url}，错误：{str(e)}")
            return ""

    def generate_summary(self, file_name: str, doxygen_content: str) -> str:
        """基于Doxygen注释生成功能摘要"""
        if not doxygen_content:
            return "Doxygen注释内容为空，无法生成摘要"

        prompt = f"""
        基于LLVM源码文件 {file_name} 的Doxygen注释，生成功能摘要需满足：
        1. 聚焦源码注释中明确的功能：该文件/模块的核心作用、关键类/函数功能、涉及的编译阶段。
        2. 去除参数说明、返回值描述、代码实现细节等非功能信息。
        3. 长度控制在100字以内，自然语言清晰表达。
        4. 简要提及设计目标或模块交互（若有）。

        Doxygen注释内容：{doxygen_content}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            with self.print_lock:
                print(f"[摘要生成失败] 文件：{file_name}，错误：{str(e)}")
            return ""

    def process_single_file(self, file_name: str, url: str) -> Tuple[str, str]:
        """单文件处理逻辑：缓存检查→内容提取→摘要生成"""
        cache_path = os.path.join(self.cache_dir, f"{file_name}.json")

        # 优先加载缓存（避免重复处理）
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                summary = json.load(f)["summary"]
            with self.print_lock:
                print(f"[加载缓存] {file_name}")
            return file_name, summary

        # 无缓存时处理
        with self.print_lock:
            print(f"[处理中] {file_name}（线程：{threading.current_thread().name}）")

        doxygen_content = self.fetch_doxygen_content(url)
        summary = self.generate_summary(file_name, doxygen_content)

        # 缓存结果
        with open(cache_path, "w") as f:
            json.dump({
                "file_name": file_name,
                "doxygen_url": url,
                "summary": summary,
                "model": self.llm_model
            }, f, indent=2, ensure_ascii=False)

        return file_name, summary

    def run(self, target_files: List[str] = None):
        """多线程完整流程：爬取链接→并行处理→汇总结果"""
        # 步骤1：串行爬取所有Doxygen链接（仅一次IO操作，无需并行）
        file_url_map = self.crawl_llvm_doxygen_links()
        if not file_url_map:
            return

        # 步骤2：筛选目标文件（可选）
        if target_files:
            file_url_map = {f: u for f, u in file_url_map.items() if f in target_files}
        if not file_url_map:
            with self.print_lock:
                print("无匹配的目标文件，终止流程")
            return

        # 步骤3：多线程并行处理文件
        summaries = {}
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="LLVM-Summary") as executor:
            # 提交所有任务（文件名为任务标识）
            future_to_file = {
                executor.submit(self.process_single_file, file_name, url): (file_name, url)
                for file_name, url in file_url_map.items()
            }

            # 收集完成的任务结果
            for future in as_completed(future_to_file):
                file_name, _ = future_to_file[future]
                try:
                    file_name, summary = future.result()
                    summaries[file_name] = summary
                except Exception as e:
                    with self.print_lock:
                        print(f"[任务失败] 文件：{file_name}，错误：{str(e)}")

        # 步骤4：统一输出结果（避免多线程打印混乱）
        with self.print_lock:
            print("\n===== LLVM源码Doxygen文档功能摘要（多线程处理完成） =====")
            for file_name, summary in sorted(summaries.items()):  # 按文件名排序输出
                print(f"\n【{file_name}】\n功能摘要：{summary}")
            print(f"\n总计处理 {len(summaries)}/{len(file_url_map)} 个文件")


if __name__ == "__main__":
    # 初始化：设置最大线程数为8（可根据网络带宽和API并发限制调整）
    summarizer = LLVMAutoSummary(llm_model="deepseek-chat", max_workers=8)
    # 运行（如需指定目标文件，添加 target_files 参数，例：target_files=["DAGCombiner.cpp"]）
    summarizer.run()