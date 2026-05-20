import streamlit as st
from dataclasses import replace
from pathlib import Path
from src.pipeline import Pipeline, max_config
from src.questions_processing import QuestionsProcessor
import json

# 你可以让 root_path 固定，也可以让用户输入
root_path = Path("data/stock_data")
run_config = replace(max_config, llm_reranking=True, parallel_requests=1, answering_model="gpt-4o-mini-2024-07-18")
pipeline = Pipeline(root_path, run_config=run_config)

def _normalize_answer(answer):
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except Exception:
            return {
                "step_by_step_analysis": "",
                "reasoning_summary": "",
                "relevant_pages": [],
                "references": [],
                "final_answer": answer,
            }

    if not isinstance(answer, dict):
        return {
            "step_by_step_analysis": "",
            "reasoning_summary": "",
            "relevant_pages": [],
            "references": [],
            "final_answer": answer,
        }

    if "step_by_step_analysis" in answer and "final_answer" in answer:
        return {
            "step_by_step_analysis": answer.get("step_by_step_analysis", ""),
            "reasoning_summary": answer.get("reasoning_summary", ""),
            "relevant_pages": answer.get("relevant_pages", []),
            "references": answer.get("references", []),
            "final_answer": answer.get("final_answer", ""),
        }

    content = answer.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except Exception:
            content = None

    if isinstance(content, dict):
        if "step_by_step_analysis" in content and "final_answer" in content:
            return {
                "step_by_step_analysis": content.get("step_by_step_analysis", ""),
                "reasoning_summary": content.get("reasoning_summary", ""),
                "relevant_pages": content.get("relevant_pages", []),
                "references": content.get("references", []),
                "final_answer": content.get("final_answer", ""),
            }

        inner = content.get("final_answer")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = None
        if isinstance(inner, dict) and "final_answer" in inner:
            return {
                "step_by_step_analysis": inner.get("step_by_step_analysis", ""),
                "reasoning_summary": inner.get("reasoning_summary", ""),
                "relevant_pages": inner.get("relevant_pages", []),
                "references": inner.get("references", []),
                "final_answer": inner.get("final_answer", ""),
            }

    final_answer = answer.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip().startswith("{"):
        try:
            parsed = json.loads(final_answer)
            if isinstance(parsed, dict) and "final_answer" in parsed:
                return {
                    "step_by_step_analysis": parsed.get("step_by_step_analysis", ""),
                    "reasoning_summary": parsed.get("reasoning_summary", ""),
                    "relevant_pages": parsed.get("relevant_pages", []),
                    "references": parsed.get("references", []),
                    "final_answer": parsed.get("final_answer", ""),
                }
        except Exception:
            pass

    return {
        "step_by_step_analysis": answer.get("step_by_step_analysis", ""),
        "reasoning_summary": answer.get("reasoning_summary", ""),
        "relevant_pages": answer.get("relevant_pages", []),
        "references": answer.get("references", []),
        "final_answer": answer.get("final_answer", ""),
    }

st.set_page_config(page_title="RAG Challenge 2", layout="wide")

# 页面标题
st.markdown("""
<div style='background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%); padding: 20px 0; border-radius: 12px; text-align: center;'>
    <h2 style='color: white; margin: 0;'>🚀 RAG Challenge 2</h2>
    <div style='color: #fff; font-size: 16px;'>基于深度RAG系统，由RTX 5080 GPU加速 | 支持多公司年报问答 | 向量检索+LLM推理+GPT-4o</div>
</div>
""", unsafe_allow_html=True)

# 左侧输入区
with st.sidebar:
    st.header("查询设置")
    # 仅单问题输入
    user_question = st.text_area("输入问题", "请简要总结公司2022年主营业务的主要内容。", height=80)
    submit_btn = st.button("生成答案", use_container_width=True)

# 右侧主内容区
st.markdown("<h3 style='margin-top: 24px;'>检索结果</h3>", unsafe_allow_html=True)

if submit_btn and user_question.strip():
    with st.spinner("正在生成答案，请稍候..."):
        try:
            answer = pipeline.answer_single_question(user_question, kind="string")
            normalized = _normalize_answer(answer)
            step_by_step = normalized.get("step_by_step_analysis", "-") or "-"
            reasoning_summary = normalized.get("reasoning_summary", "-") or "-"
            relevant_pages = normalized.get("relevant_pages", [])
            references = normalized.get("references", [])
            final_answer = normalized.get("final_answer", "-")
            # 打印调试
            print("[DEBUG] step_by_step_analysis:", step_by_step)
            print("[DEBUG] reasoning_summary:", reasoning_summary)
            print("[DEBUG] relevant_pages:", relevant_pages)
            print("[DEBUG] final_answer:", final_answer)
            st.markdown("**分步推理：**")
            st.info(step_by_step)
            st.markdown("**推理摘要：**")
            st.success(reasoning_summary)
            st.markdown("**相关页面：** ")
            st.write(relevant_pages)
            st.markdown("**引用详情：**")
            if references:
                for ref in references:
                    st.markdown(
                        f"- `{ref.get('pdf_sha1', '')}` | `{ref.get('document_name', '')}` | `{ref.get('page', f'page {ref.get('page_index', '')}')}`"
                    )
            else:
                st.write("暂无引用详情")
            st.markdown("**最终答案：**")
            st.markdown(f"<div style='background:#f6f8fa;padding:16px;border-radius:8px;font-size:18px;'>{final_answer}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"生成答案时出错: {e}")
else:
    st.info("请在左侧输入问题并点击【生成答案】") 
