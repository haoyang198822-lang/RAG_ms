import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
import requests
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt


# BM25Ingestor：BM25索引构建与保存工具
class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """从文本块列表创建BM25索引"""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """
        批量处理所有报告，生成并保存BM25索引。
        参数：
            all_reports_dir (Path): 存放JSON报告的目录
            output_dir (Path): 保存BM25索引的目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # 提取文本块并创建BM25索引
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # 保存BM25索引，文件名用sha1_name
            sha1_name = report_data["metainfo"]["sha1"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

# VectorDBIngestor：向量库构建与保存工具
class VectorDBIngestor:
    def __init__(self):
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
        self.api_key = os.getenv("AGICTO_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("AGICTO_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.agicto.cn/v1"
        self.embedding_model = os.getenv("AGICTO_EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-v4"

    def _parse_embedding_response(self, response_json):
        if isinstance(response_json, dict):
            if isinstance(response_json.get("data"), list) and response_json["data"]:
                first_item = response_json["data"][0]
                if isinstance(first_item, dict) and "embedding" in first_item:
                    return [item["embedding"] for item in response_json["data"] if item.get("embedding")]
            if isinstance(response_json.get("output"), dict):
                output = response_json["output"]
                if isinstance(output.get("embeddings"), list) and output["embeddings"]:
                    embeddings = []
                    for item in output["embeddings"]:
                        if isinstance(item, dict) and item.get("embedding"):
                            embeddings.append(item["embedding"])
                    if embeddings:
                        return embeddings
                if isinstance(output.get("embedding"), list) and output["embedding"]:
                    return [output["embedding"]]
            if isinstance(response_json.get("embedding"), list) and response_json["embedding"]:
                return [response_json["embedding"]]
        raise ValueError(f"无法解析 embedding 返回格式: {response_json}")

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-v4") -> List[float]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        if isinstance(text, list):
            text_chunks = text
        else:
            text_chunks = [text]

        if not all(isinstance(x, str) for x in text_chunks):
            raise ValueError("所有待嵌入文本必须为字符串类型！实际类型: {}".format([type(x) for x in text_chunks]))

        text_chunks = [x for x in text_chunks if x.strip()]
        if not text_chunks:
            raise ValueError("所有待嵌入文本均为空字符串！")

        print('start embedding ================================')
        embeddings = []
        MAX_BATCH_SIZE = 10
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for i in range(0, len(text_chunks), MAX_BATCH_SIZE):
            batch = text_chunks[i:i + MAX_BATCH_SIZE]
            payload = {
                "model": model or self.embedding_model,
                "input": batch,
            }
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/embeddings",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            response_json = resp.json()
            batch_embeddings = self._parse_embedding_response(response_json)
            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    f"embedding 返回条数与输入不一致，input={len(batch)}，output={len(batch_embeddings)}，response={response_json}"
                )
            for idx, emb in enumerate(batch_embeddings):
                if emb is None or len(emb) == 0:
                    raise RuntimeError(f"Embedding返回为空，batch_index={i + idx}")
                embeddings.append(emb)
        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        # 用faiss构建向量库，采用内积（余弦距离）
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        # 针对单份报告，提取文本块并生成向量库
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        # 过滤空内容，超长内容截断到 2048 字符
        max_len = 2048
        text_chunks = [t[:max_len] for t in text_chunks if len(t) > 0]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        # 批量处理所有报告，生成并保存faiss向量库
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports for FAISS"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            index = self._process_report(report_data)
            # 用 metainfo['sha1'] 作为 faiss 文件名，避免中文和特殊字符
            sha1 = report_data["metainfo"].get("sha1", "")
            if not sha1:
                raise ValueError(f"分块报告 {report_path} 缺少 sha1 字段，无法保存 faiss 文件！")
            faiss_file_path = output_dir / f"{sha1}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")