import os
import redis
from datetime import datetime

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_MODEL_REGISTRY = {
    # Cheapest input
    "stepfun": "stepfun/step-3.7-flash",

    # Cheapest output / strong default for coding and agents
    "deepseek": "deepseek/deepseek-v4-pro",

    # Cheap multimodal / long-context alternative
    "minimax": "minimax/minimax-m3",

    # More expensive alternatives
    "kimi": "moonshotai/kimi-k2.6",
    "zai": "z-ai/glm-5.1",
    "qwen": "qwen/qwen3.7-max",
    "gemini": "google/gemini-3.5-flash",
    "anthropic": "anthropic/claude-opus-4.6",
    "openai": "openai/gpt-5.5",
}
MODEL_TYPE = "openai"

def time_dir():
    now = datetime.now()
    return os.path.join(
        f"{now.year}-{now.month}",f"{now.day}"
    )

class Config:
    LLM_CALL_PASSWORD = "BioAgent"
    LLM_CALL_WAITING_TIME = 1
    LLM_SERVER_IP = ""

    BASE_DIR = Path(__file__).resolve().parents[0]
    LOG_DIR = os.path.join(BASE_DIR,"log")
    TASK_DIR = os.path.join(BASE_DIR,"task")
    TOOL_DOC_DIR = os.path.join(BASE_DIR,"tool/doc")
    TOOL_CODE_DIR = os.path.join(BASE_DIR, "tool/code")

    ZIP_DIR = "zip"

    SAVE_LOG = True
    ECHO_INFO = True

    SAVE_MEMORY = True
    MEMORY_PREFIX = "v1.0.0"

    USE_MEMORY = True
    USE_FILE_APPENDIX = False #True的话可以吧文件内容当作提示词输入到模型中，不过会进行截断


    BASE_LLM_MODEL = OPENROUTER_MODEL_REGISTRY[MODEL_TYPE]
    SUPER_LLM_MODEL = OPENROUTER_MODEL_REGISTRY[MODEL_TYPE]

    HIGHSCORE_TOOL_THRESHOLD = 5
    WORKFLOW_USED_TOOL_THRESHOLD = 5

    EXECUTOR_CODE_WAITING_TIME = 1
    ACTION_RETRY_TIMES = 4

    #######################################
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379

    REDIS_GPT_TASK_KEY = "bioagent:gpt:task"
    REDIS_GPT_RESULT_KEY = "bioagent:gpt:result"


    REDIS_LLAMA_TASK_KEY = "bioagent:llama:task"
    REDIS_LLAMA_RESULT_KEY = "bioagent:llama:result"

    REDIS_EXECUTOR_LIST_TASK_KEY = "bioagent:executor:task"
    REDIS_EXECUTOR_LIST_DATA_KEY = "bioagent:executor:data"

    REDIS_ACTIVE_TOOL_KEY = "bioagent:tools:active"
    REDIS_TOOL_INFO_KEY = "bioagent:tools:info"
    REDIS_STATUS_DATA_KEY = "bioagent:status:data"
    REDIS_STATUS_LIST_KEY = "bioagent:status:list"
    REDIS_PROGRESS_DATA_KEY = "bioagent:progress_data"
    REDIS_RESULT_DATA_KEY = "bioagent:result_data"

    REDIS_INSTANCE_INFO_KEY = "bioagent:instance:info"
    REDIS_INSTANCE_PUBLISH_KEY = "bioagent:instance:publish"

    REDIS_MEMORY_STORAGE_KEY = f"bioagent:memory:{MEMORY_PREFIX}:storage"
    REDIS_MEMORY_INFO_KEY = f"bioagent:memory:info"
    REDIS_MEMORY_LOG = f"bioagent:memory:{MEMORY_PREFIX}:log:"
    REDIS_MEMORY_LOG2 = f"bioagent:memory:{MEMORY_PREFIX}:log"

    def __init__(self, task_id=None) -> None:
        self.task_id = task_id
        self.time_path = time_dir()

    def get_task(self):
        if self.task_id is None:
            raise ValueError("task id is None")
        return f"{self.task_id}"
    
    def get_redis_connect(self):
        return redis.Redis(self.REDIS_HOST,self.REDIS_PORT,decode_responses=True)
    
    @classmethod
    def set_task(cls,task_id):
        return Config(task_id)
    
    def log_error(self, item:str, message:str):
        time = datetime.now()
        with open("BioLog/error.log","a",encoding="utf8") as f:
            f.write(f"{item=}\n{time=}\n{message=}\n{'='*50}\n")


config = Config()