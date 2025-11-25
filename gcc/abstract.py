import requests
from bs4 import BeautifulSoup
import os
import json
from typing import Dict, List, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class GCCAutoSummary:
    def __init__(self, llm_model: str = "deepseek-chat",
                 cache_dir: str = "/home/chris/FLL-workplace/gcc_summaries",
                 max_workers: int = 8):
        """
        初始化GCC源码Doxygen文档摘要生成器（爬取官网文档）
        :param llm_model: 大模型名称（默认deepseek-chat）
        :param cache_dir: 缓存目录（存储生成的摘要）
        :param max_workers: 最大线程数（控制并发爬取和生成效率）
        """
        self.llm_model = llm_model
        self.cache_dir = cache_dir
        self.max_workers = max_workers

        # 创建缓存目录（不存在则创建）
        os.makedirs(cache_dir, exist_ok=True)

        # 初始化DeepSeek客户端（用户指定配置）
        self.client = OpenAI(
            api_key="sk-ad1a7b32b3f2419db17ed342a23b6b06",
            base_url="https://api.deepseek.com/v1"
        )

        # 线程锁：避免多线程打印冲突和文件写入竞争
        self.print_lock = threading.Lock()
        self.cache_lock = threading.Lock()

        # GCC Doxygen官网基础配置（参考论文中GCC文档链接格式）
        self.base_url = "https://gcc.opensuse.org/gcc-doxygen/"
        self.file_index_url = f"{self.base_url}files.html"  # GCC文件列表页
        # GCC文档后缀映射（官网Doxygen文件命名规则）
        self.ext_map = {
            "_8cc.html": ".cc",
            "_8cpp.html": ".cpp",
            "_8h.html": ".h",
            "_8c.html": ".c",
            "_8hh.html": ".hh"
        }

    def crawl_gcc_doxygen_links(self) -> Dict[str, str]:
        """
        爬取GCC官网Doxygen文档的文件链接（串行执行，仅一次）
        返回：{文件名: 文档URL} 映射
        """
        file_url_map = {}
        try:
            with self.print_lock:
                print(f"开始爬取GCC官网Doxygen文件列表：{self.file_index_url}")

            # 发送请求获取文件列表页
            response = requests.get(self.file_index_url, timeout=15)
            response.raise_for_status()  # 触发HTTP错误（如404、500）
            soup = BeautifulSoup(response.text, "html.parser")

            # 遍历所有<a>标签，筛选GCC源码文件对应的文档链接
            for link in soup.find_all("a", href=True):
                href = link["href"]
                # 过滤条件：包含目标后缀 + 非外部链接 + 非目录链接
                if any(ext in href for ext in self.ext_map.keys()) and not href.startswith("http"):
                    # 还原原始文件名（如 tree-ssa-threadupdate_8cc.html → tree-ssa-threadupdate.cc）
                    raw_file_name = href
                    for ext_html, ext_src in self.ext_map.items():
                        raw_file_name = raw_file_name.replace(ext_html, ext_src)

                    # 构建完整文档URL
                    full_url = f"{self.base_url.rstrip('/')}/{href.lstrip('/')}"
                    file_url_map[raw_file_name] = full_url

            with self.print_lock:
                print(f"爬取完成！共获取 {len(file_url_map)} 个GCC源码文件的Doxygen文档链接")

        except requests.exceptions.Timeout:
            with self.print_lock:
                print(f"错误：GCC文件列表页请求超时（URL：{self.file_index_url}）")
        except requests.exceptions.RequestException as e:
            with self.print_lock:
                print(f"错误：爬取GCC文件列表失败 - {str(e)}")
        except Exception as e:
            with self.print_lock:
                print(f"未知错误：爬取GCC链接时异常 - {str(e)}")

        return file_url_map

    def fetch_doxygen_content(self, url: str) -> str:
        """
        从GCC官网Doxygen文档页面提取核心注释内容
        :param url: 文档页面URL
        :return: 清洗后的核心注释文本（限制15000字符，避免LLM输入溢出）
        """
        try:
            response = requests.get(url, timeout=20)  # 多线程下延长超时时间
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # 提取GCC Doxygen文档的核心内容块（适配官网HTML结构）
            content_blocks = []
            # 1. 优先提取class="textblock"的段落（主要注释区域）
            text_blocks = soup.find_all("div", class_="textblock")
            if text_blocks:
                content_blocks.extend(text_blocks)
            # 2. 补充提取id="details"的区域（详细说明部分）
            details_block = soup.find(id="details")
            if details_block:
                content_blocks.append(details_block)
            # 3. 最后提取class="contents"的区域（兼容部分页面结构）
            contents_block = soup.find("div", class_="contents")
            if contents_block:
                content_blocks.append(contents_block)

            # 清洗内容：去除空白、合并文本
            raw_content = "\n".join(
                [block.get_text(strip=True) for block in content_blocks if block.get_text(strip=True)]
            )

            # 限制长度（LLM输入长度限制）
            if len(raw_content) > 15000:
                raw_content = raw_content[:15000]
                with self.print_lock:
                    print(f"警告：{url} 文档内容过长，已截断至15000字符")

            return raw_content

        except requests.exceptions.Timeout:
            with self.print_lock:
                print(f"错误：提取内容超时 - URL：{url}")
        except requests.exceptions.RequestException as e:
            with self.print_lock:
                print(f"错误：获取文档失败 - URL：{url}，原因：{str(e)}")
        except Exception as e:
            with self.print_lock:
                print(f"未知错误：提取 {url} 内容时异常 - {str(e)}")

        return ""

    def generate_summary(self, file_name: str, doxygen_content: str) -> str:
        """
        基于GCC Doxygen注释生成功能摘要（参考论文AutoCBI的摘要要求）
        :param file_name: 源码文件名（含路径）
        :param doxygen_content: 提取的Doxygen注释内容
        :return: 100字以内的功能摘要
        """
        if not doxygen_content:
            return "Doxygen注释内容为空，无法生成功能摘要"

        # prompt设计：参考论文3.1节，聚焦核心功能、模块定位、编译阶段
        prompt = f"""
        基于GCC源码文件 {file_name} 的Doxygen注释，生成功能摘要需满足：
        1. 核心功能：明确文件/模块的核心作用（如语法分析、优化、代码生成等）；
        2. 模块定位：结合文件名推测所属编译阶段（前端/中端/后端）或子系统；
        3. 关键信息：简要提及核心函数/结构体的作用（若有）；
        4. 格式要求：长度≤100字，自然语言清晰，去除参数、返回值等细节。

        Doxygen注释内容：{doxygen_content}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # 低温度保证结果稳定性
                max_tokens=250,  # 限制输出长度
                timeout=10
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            with self.print_lock:
                print(f"错误：生成 {file_name} 摘要失败 - {str(e)}")
            return ""

    def process_single_file(self, file_name: str, url: str) -> Tuple[str, str]:
        """
        单文件处理逻辑：缓存检查 → 文档爬取 → 内容提取 → 摘要生成 → 缓存保存
        :param file_name: 源码文件名
        :param url: 文档URL
        :return: (文件名, 功能摘要)
        """
        # 构建缓存文件路径（用文件名的哈希值避免特殊字符问题）
        safe_file_name = file_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        cache_path = os.path.join(self.cache_dir, f"{safe_file_name}.json")

        # 优先读取缓存（避免重复爬取和生成）
        if os.path.exists(cache_path):
            with self.cache_lock:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    with self.print_lock:
                        print(f"[缓存加载] {file_name}")
                    return file_name, cache_data["summary"]
                except Exception as e:
                    with self.print_lock:
                        print(f"警告：读取 {file_name} 缓存失败，将重新处理 - {str(e)}")

        # 无缓存时执行完整流程
        with self.print_lock:
            print(f"[处理中] {file_name}（线程：{threading.current_thread().name}）")

        # 1. 爬取并提取Doxygen内容
        doxygen_content = self.fetch_doxygen_content(url)
        if not doxygen_content:
            return file_name, "文档内容提取失败，无法生成摘要"

        # 2. 生成功能摘要
        summary = self.generate_summary(file_name, doxygen_content)
        if not summary:
            return file_name, "摘要生成失败"

        # 3. 保存缓存（线程安全）
        with self.cache_lock:
            try:
                cache_data = {
                    "file_name": file_name,
                    "doxygen_url": url,
                    "summary": summary,
                    "model": self.llm_model,
                }
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                with self.print_lock:
                    print(f"警告：保存 {file_name} 缓存失败 - {str(e)}")

        return file_name, summary

    def run(self, target_files: List[str] = None):
        """
        完整流程执行：爬取链接 → 筛选目标文件 → 多线程处理 → 结果输出
        :param target_files: 目标文件列表（如["tree-ssa-threadupdate.cc"]），None则处理所有文件
        """
        # 步骤1：爬取所有GCC Doxygen文档链接
        file_url_map = self.crawl_gcc_doxygen_links()
        if not file_url_map:
            with self.print_lock:
                print("没有获取到任何GCC文档链接，终止流程")
            return

        # 步骤2：筛选目标文件（可选）
        if target_files:
            filtered_map = {}
            for file_name, url in file_url_map.items():
                # 支持精确匹配和后缀匹配（如["*.cc"]匹配所有.cc文件）
                if any(
                        (target == file_name) or
                        (target.startswith("*.") and file_name.endswith(target[2:]))
                        for target in target_files
                ):
                    filtered_map[file_name] = url
            file_url_map = filtered_map
            with self.print_lock:
                print(f"筛选后剩余 {len(file_url_map)} 个目标文件")
        if not file_url_map:
            with self.print_lock:
                print("没有匹配的目标文件，终止流程")
            return

        # 步骤3：多线程并行处理文件
        summaries = {}
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="GCC-Summary") as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.process_single_file, file_name, url): file_name
                for file_name, url in file_url_map.items()
            }

            # 收集任务结果
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    file_name, summary = future.result()
                    summaries[file_name] = summary
                except Exception as e:
                    with self.print_lock:
                        print(f"错误：处理 {file_name} 时任务失败 - {str(e)}")

        # 步骤4：统一输出结果（按文件名排序）
        with self.print_lock:
            print("\n" + "=" * 80)
            print("GCC源码文件功能摘要（多线程处理完成）")
            print("=" * 80)
            for file_name, summary in sorted(summaries.items()):
                print(f"\n【文件】{file_name}")
                print(f"【功能摘要】{summary}")
            print(f"\n总计处理：{len(summaries)}/{len(file_url_map)} 个文件")
            print(f"缓存目录：{self.cache_dir}")
            print("=" * 80)


if __name__ == "__main__":
    # 初始化生成器（可调整max_workers，建议根据网络带宽调整，避免触发官网反爬）
    gcc_summarizer = GCCAutoSummary(
        llm_model="deepseek-chat",
        cache_dir="C:/Users/AA/Desktop/实验结果/gcc_doxygen_summaries",
        max_workers=4  # GCC官网建议降低并发，避免被限制访问
    )

    # 运行方式1：处理所有爬取到的文件
    # gcc_summarizer.run()

    # 运行方式2：处理指定文件（支持精确匹配和后缀匹配）
    gcc_summarizer.run(target_files=[
        "*.h",  # 匹配所有.h头文件
        "*.cc"  # 匹配所有.cc源文件
    ])
