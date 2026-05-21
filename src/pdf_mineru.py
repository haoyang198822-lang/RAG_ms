import hashlib
import os
import requests
import time
import zipfile
from pathlib import Path

api_key = os.getenv("MINERU_API_KEY", "")
BASE_URL = "https://mineru.net/api/v4"


def _auth_header():
    if not api_key:
        raise RuntimeError("MINERU_API_KEY 未设置，请先在环境变量中配置 MinerU token")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def get_task_id(file_name):
    """通过远程文件 URL 提交解析任务。"""
    url = f"{BASE_URL}/extract/task"
    header = _auth_header()
    pdf_url = "https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/" + file_name
    data = {
        "url": pdf_url,
        "is_ocr": True,
        "enable_formula": False,
    }

    res = requests.post(url, headers=header, json=data)
    print(res.status_code)
    print(res.json())
    print(res.json().get("data"))
    task_data = res.json().get("data") or {}
    task_id = task_data.get("task_id")
    if not task_id:
        raise RuntimeError(f"MinerU 任务创建失败: {res.text}")
    return task_id


def get_task_id_from_local_file(file_path, is_ocr=True, enable_formula=False, enable_table=True, language="ch"):
    """通过本地文件上传方式创建 MinerU 解析任务。"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    url = f"{BASE_URL}/file-urls/batch"
    header = _auth_header()
    short_hash = hashlib.md5(file_path.stem.encode("utf-8")).hexdigest()[:8]
    data_id = f"{file_path.stem[:80]}_{short_hash}"
    data = {
        "files": [
            {
                "name": file_path.name,
                "data_id": data_id,
                "is_ocr": is_ocr,
            }
        ],
        "model_version": "vlm",
        "enable_formula": enable_formula,
        "enable_table": enable_table,
        "language": language,
    }

    res = requests.post(url, headers=header, json=data)
    print(res.status_code)
    result = res.json()
    print(result)
    if result.get("code") != 0:
        raise RuntimeError(f"申请上传链接失败: {result}")

    batch_id = result["data"]["batch_id"]
    file_urls = result["data"]["file_urls"]
    if not file_urls:
        raise RuntimeError("未获取到上传链接")

    upload_res = requests.put(file_urls[0], data=file_path.open("rb"))
    if upload_res.status_code not in (200, 201):
        raise RuntimeError(f"文件上传失败: HTTP {upload_res.status_code}, {upload_res.text}")

    return batch_id


def get_result(task_id):
    """查询单任务结果并下载解压。"""
    url = f"{BASE_URL}/extract/task/{task_id}"
    header = _auth_header()

    while True:
        res = requests.get(url, headers=header)
        result = res.json()["data"]
        print(result)
        state = result.get("state")
        err_msg = result.get("err_msg", "")
        if state in ["pending", "running", "converting"]:
            print("任务未完成，等待5秒后重试...")
            time.sleep(5)
            continue
        if err_msg:
            print(f"任务出错: {err_msg}")
            return
        if state == "done":
            full_zip_url = result.get("full_zip_url")
            if full_zip_url:
                local_filename = f"{task_id}.zip"
                print(f"开始下载: {full_zip_url}")
                r = requests.get(full_zip_url, stream=True)
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"下载完成，已保存到: {local_filename}")
                unzip_file(local_filename)
            else:
                print("未找到 full_zip_url，无法下载。")
            return
        print(f"未知状态: {state}")
        return


def get_result_by_batch_id(batch_id):
    """查询批量任务结果并下载第一个完成任务的压缩包。"""
    url = f"{BASE_URL}/extract-results/batch/{batch_id}"
    header = _auth_header()

    while True:
        res = requests.get(url, headers=header)
        result = res.json()["data"]
        print(result)
        extract_result = result.get("extract_result", [])
        if not extract_result:
            print("批量任务还未返回结果，等待5秒后重试...")
            time.sleep(5)
            continue

        done_one = None
        pending_states = {"waiting-file", "pending", "running", "converting"}
        for item in extract_result:
            if item.get("state") in pending_states:
                done_one = None
                break
            if item.get("state") == "done":
                done_one = item
                break

        if done_one is None:
            if any(item.get("state") in pending_states for item in extract_result):
                print("批量任务未完成，等待5秒后重试...")
                time.sleep(5)
                continue
            print("批量任务无可下载结果。")
            return

        full_zip_url = done_one.get("full_zip_url")
        if full_zip_url:
            task_name = done_one.get("file_name", batch_id)
            local_filename = f"{Path(task_name).stem}.zip"
            print(f"开始下载: {full_zip_url}")
            r = requests.get(full_zip_url, stream=True)
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"下载完成，已保存到: {local_filename}")
            unzip_file(local_filename)
        else:
            print("未找到 full_zip_url，无法下载。")
        return


def unzip_file(zip_path, extract_dir=None):
    """解压指定的zip文件到目标文件夹。"""
    if extract_dir is None:
        extract_dir = zip_path.rstrip('.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")


if __name__ == "__main__":
    file_name = '【财报】中芯国际：中芯国际2024年年度报告.pdf'
    task_id = get_task_id(file_name)
    print('task_id:', task_id)
    get_result(task_id)
