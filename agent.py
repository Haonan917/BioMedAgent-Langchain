from __future__ import annotations
from scripts.component import Task
import re
from typing import Any, Callable, Mapping, Optional, Literal

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config
from pydantic import BaseModel, Field
from scripts.prompt import *
from memory.memory_retriever import RedisMemoryRetriever
    
import asyncio
import json
# from langchain.chat_models import init_chat_model

conf = Config()
memory_retriever = RedisMemoryRetriever()


class BaseAgent:
    system = "system prompt"

    actions_template = {
        "initial":{
            "prompt":"initial prompt is {prompt}",
            "keywords":["prompt",]
        },
        "someone":{
            "prompt":"{value} is good",
            "keywords":["value",]
        }
    }
    
    def __init__(self, task:Task) -> None:
        self.task = task
        self.config = task.config
        self.model_name = self.config.BASE_LLM_MODEL
        self.temperature = 0.2        
        
    def _status_get(self, key: str, default: Any = None) -> Any:
        status = self.task.status

        if hasattr(status, "get"):
            value = status.get(key)
            if value is not None:
                return value

        return getattr(status, key, default)
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        同时兼容：
        1. dict config
        2. object config，例如 config.BASE_LLM_MODEL
        """
        if self.config is None:
            return default

        if isinstance(self.config, Mapping):
            return self.config.get(key, default)

        return getattr(self.config, key, default)
    
    def create_model(self):
        return ChatOpenRouter(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=1024,
            max_retries=2,
            openrouter_provider={"data_collection": "deny"},
            seed=42,
            # other params...
        )
    
    def get_action_template(self, action: str = "initial") -> str:
        if action not in self.actions_template:
            raise KeyError(
                f"Unknown action: {action}. "
                f"Available actions: {list(self.actions_template.keys())}"
            )

        action_spec = self.actions_template[action]

        if "prompt" not in action_spec:
            raise KeyError(f"Action {action} does not contain a 'prompt' field.")

        return action_spec["prompt"]
    
    def build_prompt(
        self,
        action: str = "initial",
        system: Optional[str] = None,
    ) -> ChatPromptTemplate:
        system_prompt = system or self.system
        user_prompt = self.get_action_template(action)

        return ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessagePromptTemplate.from_template(user_prompt)
            ]
        )
    def build_chain(
        self,
        action: str = "initial",
        system: Optional[str] = None,
        output_schema: Optional[type] = None,
        parser: Optional[Runnable] = None,
        check_function: Optional[Callable[[Any], bool]] = None,
        max_retry_times: int = 3,
    ) -> Runnable:
        """
        LangChain Runnable chain
            prompt | model | parser / prompt | model.with_structured_output(output_schema) / prompt | model | parser | validator
        If max_retry_times > 1:
            chain.with_retry(stop_after_attempt=max_retry_times)
        """
        self.model = self.create_model()
        prompt = self.build_prompt(action=action, system=system)

        if output_schema is not None and parser is not None:
            raise ValueError("output_schema and parser cannot be used together.")

        if output_schema is not None:
            # 推荐方式：结构化输出，替代 XML 正则解析
            model = self.model.with_structured_output(output_schema)
            chain: Runnable = prompt | model
        else:
            # 默认返回字符串
            output_parser = parser or StrOutputParser()
            chain = prompt | self.model | output_parser

        if check_function is not None:

            def validate_output(output: Any) -> Any:
                if not check_function(output):
                    raise ValueError(f"Invalid model output: {output}")
                return output

            chain = chain | RunnableLambda(validate_output)

        if max_retry_times and max_retry_times > 1:
            chain = chain.with_retry(stop_after_attempt=max_retry_times)

        return chain
    
    def invoke_action(
        self,
        action: str = "initial",
        data: Optional[dict[str, Any]] = None,
        system: Optional[str] = None,
        output_schema: Optional[type] = None,
        parser: Optional[Runnable] = None,
        check_function: Optional[Callable[[Any], bool]] = None,
        max_retry_times: int = 3,
    ) -> Any:

        chain = self.build_chain(
            action=action,
            system=system,
            output_schema=output_schema,
            parser=parser,
            check_function=check_function,
            max_retry_times=max_retry_times,
        )

        return chain.invoke(data or {})
    
    async def ainvoke_action(
        self,
        action: str = "initial",
        data: Optional[dict[str, Any]] = None,
        system: Optional[str] = None,
        output_schema: Optional[type] = None,
        parser: Optional[Runnable] = None,
        check_function: Optional[Callable[[Any], bool]] = None,
        max_retry_times: int = 3,
    ) -> Any:

        chain = self.build_chain(
            action=action,
            system=system,
            output_schema=output_schema,
            parser=parser,
            check_function=check_function,
            max_retry_times=max_retry_times,
        )

        return await chain.ainvoke(data or {})
    
    def update_task_status(self, key: str, value: Any) -> None:
        """
        可选：兼容你旧版 task.status 机制。

        如果你完全迁移到 LangGraph，这个方法可以不要。
        """
        if self.task is None:
            return

        status = getattr(self.task, "status", None)

        if status is None:
            return

        setattr(status, key, value)
        
    def as_langgraph_node(
        self,
        action: str = "initial",
        output_key: str = "result",
        input_mapper: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        system: Optional[str] = None,
        output_schema: Optional[type] = None,
        parser: Optional[Runnable] = None,
        check_function: Optional[Callable[[Any], bool]] = None,
        max_retry_times: int = 3,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        把当前 Agent 的某个 action 包装成 LangGraph node。

        LangGraph node 的推荐形式是：

            state -> partial_state

        所以这里返回：

            {"某个状态字段": result}

        这就替代了你原来的：

            self.task.status.__setattr__(key, response)
        """
        chain = self.build_chain(
            action=action,
            system=system,
            output_schema=output_schema,
            parser=parser,
            check_function=check_function,
            max_retry_times=max_retry_times,
        )

        def node(state: dict[str, Any]) -> dict[str, Any]:
            if input_mapper is not None:
                chain_input = input_mapper(state)
            else:
                chain_input = dict(state)

            result = chain.invoke(chain_input)
            return {output_key: result}

        return node
    
    
class EnglishCheckResult(BaseModel):
    result: Literal["YES", "NO"] = Field(
        description="Whether the input question is written in English."
    )

class Linguist(BaseAgent):
    system = linguist_system
    actions_template={
        "initial":{
            "prompt":linguist_prompt,
            "keyword": ["prompt"],
        }
    }
        
    def check_English_task(self) -> bool:
        question = self.task.question
        result: EnglishCheckResult = self.invoke_action(
            action="initial",
            data={"prompt": question},
            output_schema=EnglishCheckResult,
            max_retry_times=3,
        )
        is_english = result.result == "YES"
        self.update_task_status("is_English_question", is_english)
        
        return is_english


class EnglishCheckResult(BaseModel):
    result: Literal["YES", "NO"] = Field(
        description="Whether the input question is written in English."
    )

from typing import Any
from pydantic import BaseModel, Field


class TranslationResult(BaseModel):
    prompt: str = Field(
        description="The English version of the user's question. Do not answer the question."
    )


class Translator(BaseAgent):
    system = translator_system

    actions_template = {
        "initial": {
            "prompt": translator_prompt,
            "keywords": ["prompt"],
        }
    }

    def translate_question(self) -> str:
        if self.task.status.get("is_English_question"):
            translated_question = self.task.question
        else:
            result: TranslationResult = self.invoke_action(
                action="initial",
                data={
                    "prompt": self.task.question,
                },
                output_schema=TranslationResult,
                max_retry_times=3,
            )

            translated_question = result.prompt.strip()

        self.update_task_status("question", translated_question)

        return translated_question
    
class RefinedPromptResult(BaseModel):
    prompt: str = Field(
        description="The refined and optimized version of the user's question or prompt."
    )


class PromptEngineer(BaseAgent):
    system = prompt_engineer_system

    actions_template = {
        "initial": {
            "prompt": prompt_engineer_prompt,
            "keywords": ["prompt"],
        }
    }

    def __init__(self, task: Task) -> None:
        super().__init__(task)

    def refine_prompt(self) -> str:
        question = self.task.status.get("question")

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        result: RefinedPromptResult = self.invoke_action(
            action="initial",
            data={
                "prompt": question,
            },
            output_schema=RefinedPromptResult,
            max_retry_times=3,
        )

        refined_question = result.prompt.strip()

        self.update_task_status("refine_question", refined_question)

        return refined_question
    
class FileAnalysisResult(BaseModel):
    analysis: str = Field(
        description="The analysis result of the given file content with respect to the user's question."
    )

class FileAnalyst(BaseAgent):
    system = file_agent_system

    actions_template = {
        "initial": {
            "prompt": file_agent_prompt,
            "keywords": ["prompt", "file"],
        },
    }

    def __init__(self, task: Task) -> None:
        super().__init__(task)

    async def analyse_file(self, file: str) -> str:
        question = self.task.status.get("question")

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        result: FileAnalysisResult = await self.ainvoke_action(
            action="initial",
            data={
                "prompt": question,
                "file": file,
            },
            output_schema=FileAnalysisResult,
            max_retry_times=3,
        )

        return result.analysis.strip()

    async def analyse_files(self) -> dict[str, str]:
        files = getattr(self.task, "file_list", None)

        if not files:
            result = {}
            self.update_task_status("file_analysis", result)
            return result

        analyses = await asyncio.gather(
            *[
                self.analyse_file(file)
                for file in files
            ]
        )

        result = {
            file: analysis
            for file, analysis in zip(files, analyses)
        }

        self.update_task_status("file_analysis", result)

        return result


class ToolScoreResult(BaseModel):
    score: int = Field(
        ge=1,
        le=10,
        description="An integer score from 1 to 10."
    )
    reason: str = Field(
        description="A concise explanation of why this score was assigned."
    )

import asyncio
from typing import Any
from pydantic import BaseModel, Field


class ToolScoreResult(BaseModel):
    score: int = Field(description="The integer relevance score of the tool.")
    reason: str = Field(description="The reason for the score.")


class ToolScorer(BaseAgent):
    system = tool_scorer_system

    actions_template = {
        "initial": {
            "prompt": tool_scorer_prompt,
            "keywords": [
                "prompt",
                "toolname",
                "documentation",
                "memory_context",
            ],
        },
    }

    output_schema = ToolScoreResult
    score_fields = ("score", "reason")

    def __init__(self, task: Task, max_concurrency: int = 5) -> None:
        super().__init__(task=task)
        self.max_concurrency = max_concurrency
        self.temperature=0

    def _status_get(self, key: str, default: Any = None) -> Any:
        status = self.task.status

        if hasattr(status, "get"):
            value = status.get(key)
            if value is not None:
                return value

        return getattr(status, key, default)

    def _build_memory_context(self, tool: str) -> str:
        use_memory = self._get_config_value("USE_MEMORY", False)

        if not use_memory:
            return ""

        raw_question = (
            self._status_get("raw_question")
            or self._status_get("question")
            or getattr(self.task, "question", "")
        )

        task_id = self._status_get("task_id")

        memory_context = memory_retriever.match(
            tool,
            raw_question,
            task_id,
            k=5,
        )

        return memory_context or ""

    def build_extra_input(self, tool: str) -> dict[str, Any]:
        """
        给子类扩展输入用。

        ToolScorer 默认不需要额外输入。
        ToolReScorer 可以覆盖这里，加入 origin_score。
        """
        return {}

    def build_chain_input(self, tool: str) -> dict[str, Any]:
        tools = self._status_get("tools")

        if tools is None:
            raise ValueError("Missing task.status.tools.")

        if tool not in tools:
            raise KeyError(f"Tool {tool!r} not found in task.status.tools.")

        question = self._status_get("question")

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        document = tools[tool].get("document", "")
        memory_context = self._build_memory_context(tool)

        chain_input = {
            "prompt": question,
            "toolname": tool,
            "documentation": document,
            "memory_context": memory_context,
        }

        chain_input.update(self.build_extra_input(tool))

        return chain_input

    def normalize_result(self, result: BaseModel) -> dict[str, Any]:
        """
        ToolScorer 默认输出：
            {"score": int, "reason": str}

        子类可以覆盖这个方法，改成：
            {"re_score": int, "re_reason": str}
        """
        return {
            "score": int(result.score),
            "reason": result.reason.strip(),
        }

    async def tool_score(self, tool: str) -> dict[str, Any]:
        chain_input = self.build_chain_input(tool)

        result = await self.ainvoke_action(
            action="initial",
            data=chain_input,
            output_schema=self.output_schema,
            max_retry_times=3,
        )

        return self.normalize_result(result)

    async def _tool_score_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        tool: str,
    ) -> dict[str, Any]:
        async with semaphore:
            return await self.tool_score(tool)

    async def tools_score(self) -> dict[str, dict[str, Any]]:
        tools = self._status_get("tools")

        if tools is None:
            raise ValueError("Missing task.status.tools.")

        tool_names = list(tools.keys())

        semaphore = asyncio.Semaphore(self.max_concurrency)

        responses = await asyncio.gather(
            *[
                self._tool_score_with_semaphore(semaphore, tool)
                for tool in tool_names
            ]
        )

        for tool, response in zip(tool_names, responses):
            tools[tool].update(response)

        self.update_task_status("tools", tools)

        return tools

class ToolDescriptionResult(BaseModel):
    description: str = Field(
        description=(
            "A concise and accurate description of how this tool can be used "
            "to help answer the user's question."
        )
    )

class ToolDescriptor(BaseAgent):
    system = bioinformatician_system

    actions_template = {
        "initial": {
            "prompt": bioinformatician_prompt,
            "keywords": ["prompt", "toolname", "documentation"],
        },
    }

    def __init__(self, task: Task, max_concurrency: int = 5) -> None:
        super().__init__(task)
        self.max_concurrency = max_concurrency
        
    def get_need_describe_tools(self) -> list[str]:
        """
        Find tools that need a generated description.

        Logic:
            - If already has description, skip.
            - If task.status.rescored is False, use score.
            - If task.status.rescored is True, use re_score.
            - Only describe tools whose score >= HIGHSCORE_TOOL_THRESHOLD.
        """

        tools = self._status_get("tools")

        if tools is None:
            raise ValueError("Missing task.status.tools.")

        rescored = bool(self._status_get("rescored", False))
        score_key = "re_score" if rescored else "score"

        highscore_threshold = self._get_config_value(
            "HIGHSCORE_TOOL_THRESHOLD",
            0,
        )

        result = []

        for tool_name, tool_info in tools.items():
            if "description" in tool_info:
                print(f"description in tool info, skip: {tool_name}")
                continue
            score = tool_info.get(score_key)

            if score is None:
                continue

            if score >= highscore_threshold:
                result.append(tool_name)

        return result

    async def tool_describe(self, tool: str) -> str:
        tools = self._status_get("tools")

        if tools is None:
            raise ValueError("Missing task.status.tools.")

        if tool not in tools:
            raise KeyError(f"Tool {tool!r} not found in task.status.tools.")

        document = tools[tool].get("document", "")
        question = self._status_get("question")

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        result: ToolDescriptionResult = await self.ainvoke_action(
            action="initial",
            data={
                "prompt": question,
                "toolname": tool,
                "documentation": document,
            },
            output_schema=ToolDescriptionResult,
            max_retry_times=3,
        )

        return result.description.strip()


    async def _tool_describe_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        tool: str,
    ) -> str:
        async with semaphore:
            return await self.tool_describe(tool)

    async def tools_describe(self) -> dict[str, dict[str, Any]]:
        tools = self._status_get("tools")
        if tools is None:
            raise ValueError("Missing task.status.tools.")

        need_describe_tools = self.get_need_describe_tools()
        print(f"length of tools need to describe: {len(need_describe_tools)}")

        if not need_describe_tools:
            self.update_task_status("tools", tools)
            return tools

        semaphore = asyncio.Semaphore(self.max_concurrency)

        descriptions = await asyncio.gather(
            *[
                self._tool_describe_with_semaphore(semaphore, tool)
                for tool in need_describe_tools
            ]
        )

        for tool, description in zip(need_describe_tools, descriptions):
            tools[tool]["description"] = description

        self.update_task_status("tools", tools)

        return tools

class ToolReScoreResult(BaseModel):
    re_score: int = Field(
        description=(
            "The rescored integer relevance score for this tool with respect to "
            "the user's question, considering the original high-score tool suggestions."
        )
    )
    re_reason: str = Field(
        description="A concise explanation for the rescored relevance score."
    )

class ToolReScorer(ToolScorer):
    system = tool_rescorer_system

    actions_template = {
        "initial": {
            "prompt": tool_rescorer_prompt,
            "keywords": [
                "prompt",
                "origin_score",
                "toolname",
                "documentation",
                "memory_context",
            ],
        },
    }

    output_schema = ToolReScoreResult

    def __init__(self, task: Task, max_concurrency: int = 5) -> None:
        super().__init__(task=task, max_concurrency=max_concurrency)
        self._origin_score_cache: str | None = None

    def get_origin_score(self) -> str:
        tools = self._status_get("tools")

        if tools is None:
            raise ValueError("Missing task.status.tools.")

        highscore_threshold = self._get_config_value(
            "HIGHSCORE_TOOL_THRESHOLD",
            0,
        )

        result = []

        for tool_name, tool_info in tools.items():
            score = tool_info.get("score")

            if score is None:
                continue

            if score >= highscore_threshold:
                result.append(
                    {
                        "toolname": tool_name,
                        "suggestion": tool_info.get("description", ""),
                    }
                )

        return json.dumps(result, ensure_ascii=False)

    def build_extra_input(self, tool: str) -> dict[str, Any]:
        if self._origin_score_cache is None:
            self._origin_score_cache = self.get_origin_score()

        return {
            "origin_score": self._origin_score_cache
        }

    def normalize_result(self, result: ToolReScoreResult) -> dict[str, Any]:
        return {
            "re_score": int(result.re_score),
            "re_reason": result.re_reason.strip(),
        }

    async def tools_score(self) -> dict[str, dict[str, Any]]:
        self._origin_score_cache = self.get_origin_score()

        tools = await super().tools_score()

        self.update_task_status("rescored", True)

        return tools
   
   


class WorkflowDesignResult(BaseModel):
    workflow: str = Field(
        description=(
            "The final executable or actionable workflow designed for solving the user's task. "
            "It should use the selected tools appropriately and describe the steps clearly."
        )
    )


class WorkflowDesigner(BaseAgent):
    system = workflow_architect_system

    actions_template = {
        "initial": {
            "prompt": workflow_architect_prompt,
            "keywords": [
                "prompt",
                "analysis_result",
                "tool_list",
                "file_appendix",
                "memory_context",
            ],
        },
    }

    def __init__(self, task: Task) -> None:
        super().__init__(task=task)
        self.model_name = self.config.SUPER_LLM_MODEL
    def get_tools_info(self) -> list[dict[str, Any]]:
        tools = self._status_get("tools")

        if tools is None:
            raise ValueError("Missing task.status.tools.")

        workflow_threshold = self._get_config_value(
            "WORKFLOW_USED_TOOL_THRESHOLD",
            0,
        )

        rescored = bool(self._status_get("rescored", False))
        score_key = "re_score" if rescored else "score"

        tools_info = []

        for tool_name, tool_info in tools.items():
            score = tool_info.get(score_key)

            if score is None:
                continue

            if score < workflow_threshold:
                continue

            item = {
                "tool": tool_name,
                "documentation": tool_info.get("document", "")
            }

            # if "description" in tool_info:
            #     item["description"] = tool_info["description"]

            # if score_key in tool_info:
            #      item[score_key] = tool_info[score_key]
            # if "reason" in tool_info:
            #     item["reason"] = tool_info["reason"]

            # if "re_reason" in tool_info:
            #     item["re_reason"] = tool_info["re_reason"]

            tools_info.append(item)

        return tools_info

    def _build_memory_context(self) -> str:
        use_memory = self._get_config_value("USE_MEMORY", False)

        if not use_memory:
            return ""

        raw_question = (
            self._status_get("raw_question")
            or self._status_get("question")
            or getattr(self.task, "question", "")
        )

        task_id = self._status_get("task_id")

        memory_context = memory_retriever.match(
            "workflow",
            raw_question,
            task_id,
            k=3,
        )

        return memory_context or ""

    def workflow_design(self) -> str:
        question = self._status_get("question")

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        file_analysis = self._status_get("file_analysis", {})
        tools_info = self.get_tools_info()

        if not tools_info:
            raise ValueError(
                "No tools passed WORKFLOW_USED_TOOL_THRESHOLD. "
                "Please check tool scoring or lower the threshold."
            )

        file_appendix = getattr(self.task, "file_appendix", "") or ""
        memory_context = self._build_memory_context()

        result: WorkflowDesignResult = self.invoke_action(
            action="initial",
            data={
                "prompt": question,
                "analysis_result": json.dumps(file_analysis, ensure_ascii=False),
                "tool_list": json.dumps(tools_info, ensure_ascii=False),
                "file_appendix": file_appendix,
                "memory_context": memory_context,
            },
            output_schema=WorkflowDesignResult,
            max_retry_times=3,
        )

        workflow = result.workflow.strip()

        # 兼容旧状态字段
        self.update_task_status(
            "raw_workflow",
            result.model_dump_json(ensure_ascii=False),
        )
        self.update_task_status("workflow", workflow)

        return workflow
    
    

class ToolSuggestionResult(BaseModel):
    suggestion: str = Field(
        description=(
            "Concrete usage suggestion for the current tool in the designed workflow. "
            "Explain how this tool should be used, what input it needs, and what output it should produce."
        )
    )


class ToolAnalyst(BaseAgent):
    system = tool_suggestion_system

    actions_template = {
        "initial": {
            "prompt": tool_suggestion_prompt,
            "keywords": ["prompt", "workflow", "tool_info", "tool"],
        },
    }

    def __init__(self, task: Task, max_concurrency: int = 5) -> None:
        super().__init__(task=task)
        
        self.model_name = self.task.config.SUPER_LLM_MODEL
        self.max_concurrency = max_concurrency

    @staticmethod
    def _tool_mentioned_in_workflow(tool: str, workflow: str) -> bool:
        if not tool or not workflow:
            return False

        pattern = r"(?<![A-Za-z0-9_\-])" + re.escape(tool) + r"(?![A-Za-z0-9_\-])"

        return bool(re.search(pattern, workflow, re.IGNORECASE))

    def get_tool_used(self, workflow: str) -> dict[str, dict[str, Any]]:
        tools = self._status_get("tools")

        if tools is None:
            raise ValueError("Missing task.status.tools.")

        if not workflow:
            raise ValueError("Missing workflow content.")

        tool_used = {}

        for tool_name, tool_info in tools.items():
            if not self._tool_mentioned_in_workflow(tool_name, workflow):
                continue

            item = {
                "tool": tool_name,
                "documentation": tool_info.get("document", ""),
            }

            if "description" in tool_info:
                item["description"] = tool_info["description"]

            tool_used[tool_name] = item

        self.update_task_status("tool_used", tool_used)

        return tool_used

    async def tool_analyse(
        self,
        tool: str,
        workflow: str,
        tool_info: str,
    ) -> str:
        question = self._status_get("question")

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        result: ToolSuggestionResult = await self.ainvoke_action(
            action="initial",
            data={
                "prompt": question,
                "tool": tool,
                "workflow": workflow,
                "tool_info": tool_info,
            },
            output_schema=ToolSuggestionResult,
            max_retry_times=3,
        )

        return result.suggestion.strip()

    async def _tool_analyse_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        tool: str,
        workflow: str,
        tool_info: str,
    ) -> str:
        async with semaphore:
            return await self.tool_analyse(
                tool=tool,
                workflow=workflow,
                tool_info=tool_info,
            )

    async def tools_analyse(self) -> dict[str, dict[str, Any]]:
        workflow = self._status_get("workflow")

        if not workflow:
            raise ValueError("Missing task.status.workflow. Please run WorkflowDesigner first.")

        tools = self.get_tool_used(workflow)

        if not tools:
            self.update_task_status("tool_used", tools)
            return tools

        tool_info = json.dumps(tools, ensure_ascii=False)

        semaphore = asyncio.Semaphore(self.max_concurrency)

        tool_names = list(tools.keys())

        suggestions = await asyncio.gather(
            *[
                self._tool_analyse_with_semaphore(
                    semaphore=semaphore,
                    tool=tool,
                    workflow=workflow,
                    tool_info=tool_info,
                )
                for tool in tool_names
            ]
        )

        for tool, suggestion in zip(tool_names, suggestions):
            tools[tool]["suggestion"] = suggestion

        self.update_task_status("tool_used", tools)

        return tools



class WorkflowFormatResult(BaseModel):
    stages: list[str] = Field(
        description=(
            "A list of ordered workflow stages. Each stage should be a clear, "
            "self-contained step derived from the workflow."
        )
    )


class WorkflowFormatter(BaseAgent):
    system = format_desinger_system

    actions_template = {
        "initial": {
            "prompt": format_desinger_prompt,
            "keywords": ["prompt", "workflow"],
        },
    }

    def __init__(self, task: Task) -> None:
        super().__init__(task)

    def workflow_format(self) -> list[str]:
        question = self._status_get("question")
        workflow = self._status_get("workflow")

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        if not workflow:
            raise ValueError("Missing task.status.workflow. Please run WorkflowDesigner first.")

        result: WorkflowFormatResult = self.invoke_action(
            action="initial",
            data={
                "prompt": question,
                "workflow": workflow,
            },
            output_schema=WorkflowFormatResult,
            max_retry_times=3,
        )

        stages = [
            stage.strip()
            for stage in result.stages
            if stage and stage.strip()
        ]

        if not stages:
            raise ValueError("WorkflowFormatter produced no valid workflow stages.")

        self.update_task_status("workflow_stages", stages)

        return stages
    

class StageActionResult(BaseModel):
    stage: str = Field(
        description="The workflow stage being converted into executable actions."
    )
    goal: str = Field(
        description="The goal of this stage."
    )
    actions: list[str] = Field(
        description="Concrete actions required to complete this stage."
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tools used in this stage."
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="Inputs required for this stage."
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="Expected outputs of this stage."
    )


class ActionDesigner(BaseAgent):
    system = action_architecture_expert_system

    actions_template = {
        "initial": {
            "prompt": action_architecture_expert_prompt,
            "keywords": [
                "prompt",
                "workflow",
                "tool_suggestion_data",
                "stage",
            ],
        },
    }

    def __init__(self, task: Task, max_concurrency: int = 5) -> None:
        super().__init__(
            task=task,
        )
        self.max_concurrency = max_concurrency
        self.model_name = self.task.config.SUPER_LLM_MODEL

    @staticmethod
    def _stage_to_text(stage: Any) -> str:
        if isinstance(stage, str):
            return stage

        if isinstance(stage, dict):
            return json.dumps(stage, ensure_ascii=False)

        return str(stage)

    async def action_design(self, stage: Any) -> dict[str, Any]:
        question = self._status_get("question")
        workflow = self._status_get("workflow")
        tool_used = self._status_get("tool_used", {})

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        if not workflow:
            raise ValueError("Missing task.status.workflow. Please run WorkflowDesigner first.")

        if tool_used is None:
            tool_used = {}

        stage_text = self._stage_to_text(stage)

        tool_suggestion_data = json.dumps(
            tool_used,
            ensure_ascii=False,
        )

        result: StageActionResult = await self.ainvoke_action(
            action="initial",
            data={
                "prompt": question,
                "stage": stage_text,
                "workflow": workflow,
                "tool_suggestion_data": tool_suggestion_data,
            },
            output_schema=StageActionResult,
            max_retry_times=3,
        )

        return result.model_dump()

    async def _action_design_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        stage: Any,
    ) -> dict[str, Any]:
        async with semaphore:
            return await self.action_design(stage)

    async def actions_design(self) -> list[dict[str, Any]]:
        stages = self._status_get("workflow_stages")

        if not stages:
            raise ValueError("Missing task.status.workflow_stages. Please run WorkflowFormatter first.")

        semaphore = asyncio.Semaphore(self.max_concurrency)

        actions = await asyncio.gather(
            *[
                self._action_design_with_semaphore(
                    semaphore=semaphore,
                    stage=stage,
                )
                for stage in stages
            ]
        )

        self.update_task_status("actions", actions)

        return actions



class MermaidDesignResult(BaseModel):
    code: str = Field(
        description=(
            "Valid Mermaid diagram code representing the workflow. "
            "Return only the Mermaid code content, without markdown fences."
        )
    )


class MermaidDesigner(BaseAgent):
    system = mermaid_system

    actions_template = {
        "initial": {
            "prompt": mermaid_prompt,
            "keywords": ["workflow"],
        },
    }

    def __init__(self, task: Task) -> None:
        super().__init__(task=task)
        self.model_name = self.task.config.SUPER_LLM_MODEL

    @staticmethod
    def _clean_mermaid_code(code: str) -> str:

        code = code.strip()

        if code.startswith("```mermaid"):
            code = code.removeprefix("```mermaid").strip()

        if code.startswith("```"):
            code = code.removeprefix("```").strip()

        if code.endswith("```"):
            code = code.removesuffix("```").strip()

        return code

    @staticmethod
    def _validate_mermaid_code(code: str) -> None:
        valid_prefixes = (
            "graph ",
            "flowchart ",
            "sequenceDiagram",
            "stateDiagram",
            "stateDiagram-v2",
            "classDiagram",
            "gantt",
            "erDiagram",
            "journey",
            "pie",
            "mindmap",
            "timeline",
        )

        if not code.strip().startswith(valid_prefixes):
            raise ValueError(
                "Invalid Mermaid code. "
                f"Expected code to start with one of {valid_prefixes}, got: {code[:80]!r}"
            )

    def mermaid_design(self) -> str:
        workflow = self._status_get("workflow")

        if not workflow:
            raise ValueError("Missing task.status.workflow. Please run WorkflowDesigner first.")

        result: MermaidDesignResult = self.invoke_action(
            action="initial",
            data={
                "workflow": workflow,
            },
            output_schema=MermaidDesignResult,
            max_retry_times=3,
        )

        mermaid_code = self._clean_mermaid_code(result.code)
        self._validate_mermaid_code(mermaid_code)

        self.update_task_status("mermaid_code", mermaid_code)

        return mermaid_code



class WorkflowRedesignResult(BaseModel):
    workflow: str = Field(
        description=(
            "The revised workflow after considering the previous workflow, "
            "the failure suggestion, file analysis, and available tools."
        )
    )


class WorkflowReDesigner(WorkflowDesigner):
    system = re_workflow_system

    actions_template = {
        "initial": {
            "prompt": re_workflow_prompt,
            "keywords": [
                "prompt",
                "analysis_result",
                "tool_list",
                "workflow",
                "suggestion",
                "memory_context",
            ],
        },
    }

    def __init__(self, task: Task) -> None:

        super().__init__(task)

    def workflow_redesign(self, suggestion: str) -> str:
        question = self._status_get("question")
        file_analysis = self._status_get("file_analysis", {})
        previous_workflow = (
            self._status_get("raw_workflow")
            or self._status_get("workflow")
        )

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        if not previous_workflow:
            raise ValueError("Missing previous workflow. Please run WorkflowDesigner first.")

        if not suggestion:
            raise ValueError("Missing redesign suggestion.")

        tools_info = self.get_tools_info()

        if not tools_info:
            raise ValueError(
                "No tools passed WORKFLOW_USED_TOOL_THRESHOLD. "
                "Please check tool scoring or lower the threshold."
            )

        memory_context = self._build_memory_context()

        result: WorkflowRedesignResult = self.invoke_action(
            action="initial",
            data={
                "prompt": question,
                "workflow": previous_workflow,
                "suggestion": suggestion,
                "analysis_result": json.dumps(file_analysis, ensure_ascii=False),
                "tool_list": json.dumps(tools_info, ensure_ascii=False),
                "memory_context": memory_context,
            },
            output_schema=WorkflowRedesignResult,
            max_retry_times=3,
        )

        redesigned_workflow = result.workflow.strip()

        self.update_task_status(
            "raw_workflow",
            result.model_dump_json(ensure_ascii=False),
        )
        self.update_task_status("workflow", redesigned_workflow)

        return redesigned_workflow



class SummaryResult(BaseModel):
    summary: str = Field(
        description=(
            "A clear final summary of the completed workflow, generated results, "
            "important outputs, and limitations if any."
        )
    )


class SummaryAnalyst(BaseAgent):
    system = summary_system

    actions_template = {
        "initial": {
            "prompt": summary_prompt,
            "keywords": ["prompt", "workflow", "resource_pool"],
        },
    }

    def __init__(self, task: Task) -> None:

        super().__init__(
            task=task
        )
        self.model_name=model=task.config.SUPER_LLM_MODEL

    def summary(self) -> str:
        question = self._status_get("question")
        workflow = self._status_get("workflow")
        # resource_pool = self._status_get("resource_pool", [])
        resource_pool = []

        if not question:
            raise ValueError("Missing task.status.question. Please run Translator first.")

        if not workflow:
            raise ValueError("Missing task.status.workflow. Please run WorkflowDesigner first.")

        result: SummaryResult = self.invoke_action(
            action="initial",
            data={
                "prompt": question,
                "workflow": workflow,
                "resource_pool": json.dumps(resource_pool, ensure_ascii=False),
            },
            output_schema=SummaryResult,
            max_retry_times=3,
        )

        summary_text = result.summary.strip()

        self.update_task_status("summary", summary_text)

        return summary_text
    
    
    
# if __name__== "__main__":
#     system = SystemMessage(content=linguist_system)
#     human = HumanMessagePromptTemplate.from_template(linguist_prompt)

#     # 2. 组合成完整对话模板
#     chat_prompt = ChatPromptTemplate.from_messages([system, human])

#     # 3. 构建链
#     model = ChatOpenRouter(
#         model=conf.BASE_LLM_MODEL,
#         temperature=0,
#         max_tokens=1024,
#         max_retries=2,
#         openrouter_provider={"data_collection": "deny"},
#         seed=42,
#     )
#     chain: Runnable = chat_prompt | model.with_structured_output(EnglishCheckResult)

#     # 4. 调用：传入所有需要的变量
#     response = chain.invoke({'prompt': '以下是食管鳞状细胞癌数据集GSE20347中1个样本的微阵列表达数据{GSM509787_E1507N.CEL.gz}，其对应的探针编号为{GPL571}，请使用CeltoExp工具将该CEL微阵列表达文件转换为txt文件格式的基因表达谱数据。'})

#     print(type(response))  # <class 'str'>
#     print(response)  
         