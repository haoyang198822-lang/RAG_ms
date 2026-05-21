"""
API REQUEST PARALLEL PROCESSOR

本脚本用于并发处理 OpenAI API 请求，并进行速率限制控制。

功能：
- 从文件流式读取请求，适合超大批量任务，避免内存溢出
- 并发请求，最大化吞吐
- 限流（请求数、token 数），防止超额调用
- 失败自动重试，最多 {max_attempts} 次，提升鲁棒性
- 错误日志记录，便于排查问题

输入参数：
- requests_filepath : str
    - 待处理请求的 jsonl 文件路径，每行一个 json 对象，支持 metadata
    - 例：{"model": "text-embedding-3-small", "input": "embed me", "metadata": {"row_id": 1}}
- save_filepath : str, optional
    - 结果保存路径，jsonl 格式，每行为 [原始请求, API 响应, 可选 metadata]
- request_url : str, optional
    - API 接口 URL，默认 OpenAI embedding 接口
- api_key : str, optional
    - API 密钥，默认读取 OPENAI_API_KEY 环境变量
- max_requests_per_minute : float, optional
    - 每分钟最大请求数，建议留安全余量
- max_tokens_per_minute : float, optional
    - 每分钟最大 token 数，建议留安全余量
- token_encoding_name : str, optional
    - tiktoken 编码名，默认 cl100k_base
- max_attempts : int, optional
    - 单请求最大重试次数，默认 5
- logging_level : int, optional
    - 日志等级，40=ERROR，30=WARNING，20=INFO，10=DEBUG

脚本结构：
    - 导入
    - 主流程 async def process_api_requests_from_file
    - 状态追踪类 StatusTracker
    - API 请求类 APIRequest
    - 工具函数：api_endpoint_from_url、append_to_jsonl、num_tokens_consumed_from_request、task_id_generator_function
"""

# 导入
import aiohttp  # 用于并发发起 API 调用
import argparse  # 用于从命令行运行脚本
import asyncio  # 用于并发发起 API 调用
import json  # 用于将结果保存为 jsonl 文件
import logging  # 用于记录限流告警和其他信息
import os  # 用于读取 API key
import re  # 用于从请求 URL 中匹配接口
import tiktoken  # 用于统计 token 数
import time  # 用于限流后暂停
from dataclasses import (
    dataclass,
    field,
)  # 用于存储 API 输入、输出和元数据


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """并发处理 API 请求，自动限流，支持重试。"""
    # 常量
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 毫秒的循环间隔可将最大吞吐限制在每秒 1000 个请求
    )

    # 初始化日志
    logging.basicConfig(level=logging_level)
    logging.debug(f"日志已按级别 {logging_level} 初始化")

    # 推断 API 端点并构造请求头
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # Azure 部署使用 api-key 请求头
    if "/deployments" in request_url:
        request_header = {"api-key": f"{api_key}"}

    # 初始化追踪器
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # 生成 0、1、2……这样的整数 ID
    status_tracker = (
        StatusTracker()
    )  # 单例，用于追踪一组状态变量
    next_request = None  # 用于保存下一个待调用请求的变量

    # 初始化可用容量计数
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # 初始化标志位
    file_not_finished = True  # 文件读完后，将不再继续读取
    logging.debug("初始化完成。")

    # 初始化文件读取
    with open(requests_filepath) as file:
        # `requests` 会一次提供一个请求
        requests = file.__iter__()
        logging.debug("文件已打开，进入主循环")
        async with aiohttp.ClientSession() as session:  # 在此处初始化 ClientSession
            while True:
                # 获取下一个请求（如果当前没有正在等待容量的请求）
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"正在重试请求 {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # 读取新请求
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"读取请求 {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # 文件读取结束后，设置标志位停止继续读取
                            logging.debug("文件已读完")
                            file_not_finished = False

                # 更新可用容量
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # 如果容量足够，则调用 API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # 更新计数器
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # 调用 API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # 将 next_request 重置为空

                # 如果所有任务都完成，则退出
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # 主循环短暂休眠，让并发任务可以运行
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # 如果最近触发了限流错误，则暂停冷却
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # 例如，如果暂停时间为 15 秒，而最后一次限流发生在 5 秒前
                    logging.warn(
                        f"暂停冷却，直到 {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # 完成后记录最终状态
        logging.info(
            f"""并行处理完成。结果已保存到 {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} 个请求失败。错误已记录到 {save_filepath}。"
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"收到 {status_tracker.num_rate_limit_errors} 次限流错误。建议降低运行速率。"
            )


# 数据类


@dataclass
class StatusTracker:
    """存储脚本进度相关的元数据。只会创建一个实例。"""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # 当该值变为 0 时脚本结束
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # 不包含上面统计的限流错误
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # 用于在触发限流后冷却


@dataclass
class APIRequest:
    """存储 API 请求的输入、输出和其他元数据。包含一个发起 API 调用的方法。"""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """调用 OpenAI API 并保存结果。"""
        # logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"请求 {self.task_id} 失败，错误信息：{response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "").lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # 限流错误单独统计
                    )

        except (
            Exception
        ) as e:  # 这里捕获裸异常不算最佳实践，但这里会记录并保存错误
            logging.warning(f"请求 {self.task_id} 因异常失败：{e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"请求 {self.request_json} 在所有重试次数用尽后仍失败，正在保存错误：{self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"请求 {self.task_id} 已保存到 {save_filepath}")


# 函数


def api_endpoint_from_url(request_url):
    """从请求 URL 中提取 API 端点。"""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # Azure OpenAI 部署 URL 使用此模式
        match = re.search(
            r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
        )
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """将一个 json 负载追加到 jsonl 文件末尾。"""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """统计请求中的 token 数。仅支持 completion 和 embedding 请求。"""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # 如果是 completions 请求，tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # 每条消息都遵循 <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # 如果有 name，则会省略 role
                        num_tokens -= 1  # role 始终必需，并且始终占 1 个 token
            num_tokens += 2  # 每个回复都会以 <im_start>assistant 开头
            return num_tokens + completion_tokens
        # 普通 completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # 单个 prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # 多个 prompt
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # 如果是 embeddings 请求，tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # 单个输入
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # 多个输入
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # 其他 API 调用（例如 edits、inserts、DALL-E）还需要补充更多逻辑
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """生成 0、1、2 这样的整数。"""
    task_id = 0
    while True:
        yield task_id