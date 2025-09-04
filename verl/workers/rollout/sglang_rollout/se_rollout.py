# from __future__ import annotations

import asyncio
import aiohttp
# import math
# import random
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional

# import torch
# from verl import DataProto
# from verl.workers.rollout.sglang_rollout.schemas import Message
# from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj
from verl.workers.rollout.sglang_rollout import SGLangRollout
from typing import Any, List, Optional, Tuple, Dict
import json, re, os
import logging
from verl.utils.profiler import GPUMemoryLogger
import torch
from verl import DataProto
import numpy as np

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class SERollout(SGLangRollout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_conversation(self, dialog: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Parse the conversation into a structured trace.
        step 0 固定为 user 的初始问题，后续 assistant/tool 从 step 1 开始递增。
        """
        trace = []
        step = 0  

        # --- 先把 user 的初始问题写进去 ---
        for turn in dialog:
            if turn.get("role") == "user":
                text = turn.get("content", "").strip()
                if text:
                    trace.append({"step": step, "type": "user", "content": text})
                    step += 1
                break   # 只要第一条 user

        # --- 后面正常解析 assistant / tool ---
        for turn in dialog:
            role = turn.get("role")
            text = turn.get("content", "")
            tool_calls = turn.get("tool_calls")

            if role in {"system", "user"}:
                continue

            # ---------- assistant ----------
            if role == "assistant":
                # 1) think：整段内容
                if text.strip():
                    trace.append({"step": step, "type": "think", "content": text.strip()})
                    step += 1

                # 2) search
                if tool_calls:
                    for call in tool_calls:
                        search_body = json.dumps(
                            {"name": call["function"]["name"],
                            "arguments": call["function"]["arguments"]},
                            ensure_ascii=False
                        )
                        trace.append({"step": step, "type": "search", "content": search_body})
                        step += 1
                else:
                    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
                        trace.append({"step": step, "type": "search", "content": m.group(1).strip()})
                        step += 1

                # 3) answer
                m = re.search(r"<answer>(.*?)</answer>", text, re.S)
                if m:
                    trace.append({"step": step, "type": "answer", "content": m.group(1).strip()})
                    step += 1

            # ---------- tool ----------
            elif role == "tool":
                trace.append({"step": step, "type": "information", "content": text.strip()})
                step += 1

        return trace
    
    # async def _revision_one(self, traj: List[Dict[str, Any]],
    #                         sem: asyncio.Semaphore) -> List[Dict[str, Any]]:
    #     """
    #     对单条轨迹做 revision。
    #     """
    #     async with sem:                     
    #         messages = self._traj_to_messages(traj)   
    #         revised_text = await _call_llm(messages)
    #         new_traj = self._parse_conversation(
    #             [{"role": "assistant", "content": revised_text}]
    #         )
    #         return new_traj

    # def _revision(self, init_traj: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    #     """
    #     并发 revision。
    #     由于 _revision 本身是 sync 函数，用一次 run_until_complete。
    #     """
    #     async def _run():
    #         max_concurrency = 64
    #         sem = asyncio.Semaphore(max_concurrency)
    #         tasks = [self._revision_one(traj, sem) for traj in init_traj]
    #         return await asyncio.gather(*tasks)

    #     loop = asyncio.get_event_loop()
    #     return loop.run_until_complete(_run())

    def _revision(self, init_traj: List[List[Dict[str, Any]]]):
        """
        并发 revision。
        由于 _revision 本身是 sync 函数，用一次 run_until_complete。
        """
        data = [self._traj_to_messages(traj) for traj in init_traj]
        texts = self.processing_class.apply_chat_template(
            data,          
            tokenize=False,         
            add_generation_prompt=True,
        )
        model_inputs = self.processing_class(
            texts,
            padding=True,
            return_tensors="pt",    
            padding_side="left"
        )
        position_ids = model_inputs.attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(model_inputs.attention_mask == 0, 0)
        inputs = DataProto.from_dict(
            tensors = {
                "input_ids": model_inputs.input_ids,
                "attention_mask": model_inputs.attention_mask,
                "position_ids": position_ids
            },
            non_tensors = {
                "raw_prompt": np.array(data, dtype=object),
                "raw_prompt_ids": np.array(model_inputs.input_ids, dtype=object),
                "index": np.array([0] * len(data), dtype=object),
                "tools_kwargs": np.array([{"retrieve_documents": {"create_kwargs": {"": ""}}}] * len(data), dtype=object)
            }
        )
        kwargs = {"max_prompt_len": 4096}
        breakpoint()
        outputs = self._req_level_generate_sequences(inputs, **kwargs)

        return outputs


    def _traj_to_messages(self, traj: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        把结构化轨迹还原成对话格式，用于 prompt LLM。
        """
        system_prompt = '''
            You are an expert in analyzing agent trajectories.  
            Please identify the **single most decisive critical decision point** in the entire trajectory.  
            A “critical decision point” is the one step that most strongly determines whether the overall solution succeeds or fails, typically characterized by:

            - Accurately locating the core of the problem  
            - Discovering the breakthrough for the solution  
            - Making a decisive modification or judgment  
            - Establishing the correct execution path  

            You must read the whole trajectory, find this **one** most critical step, and provide:

            - **step number**  
            - **why** this step is the most critical  
            - **impact** it has on the final solution  

            Your output must strictly follow the JSON format below, containing a `critical_step` object with the following fields:

            ```json
            {
            "critical_step": {
                "step": <integer>,
                "type": "<string>",
                "reasoning": "<string>",
                "impact": "<string>"
            }
            }
        '''

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(traj)}  
        ]

        return msgs

    def _recombine():
        pass

    def _refine():
        pass

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # breakpoint()
        init_traj = self._req_level_generate_sequences(prompts, **kwargs)
        traj_formatted = []
        for m in init_traj.non_tensor_batch.get("messages", []):
            messages = [message.model_dump() for message in m["messages"]]
            traj_formatted.append(self._parse_conversation(messages))
        # breakpoint()
        revision_traj = self._revision(traj_formatted)
        recombination_traj = self._recombine(revision_traj)
        refinement_traj = self._refine(recombination_traj)

        return refinement_traj
    
    async def _call_llm(messages: List[Dict[str, str]], max_concurrency: int = 16) -> str:
        """
        虚拟 LLM：随机挑一条轨迹里的 step 作为 critical_step。
        返回合法的 JSON string，可直接被 json.loads。
        """
        # messages 的最后一个元素是用户输入的 str(traj)
        traj_str = messages[-1]["content"]
        # 简易解析：把所有 "step": <int> 抓出来
        steps = list(map(int, re.findall(r'"step":\s*(\d+)', traj_str)))
        if not steps:
            steps = [1]

        chosen = random.choice(steps)
        fake = {
            "critical_step": {
                "step": chosen,
                "type": "think",               # 可以随机挑 think/search/answer
                "reasoning": f"Step {chosen} is the breakthrough.",
                "impact": "Determines final correctness."
            }
        }
        return json.dumps(fake, ensure_ascii=False)

    # async def _call_llm(messages: List[Dict[str, str]],
    #                     max_concurrency: int = 16) -> str:
    #     """
    #     真正向 LLM 发起请求。
    #     这里以 HTTP 为例；如果直接调 SGLang 的 async_generate，同理。
    #     """
    #     # 举例：通过 HTTP 访问本地 SGLang server
    #     url = "http://localhost:8000/v1/chat/completions"

    #     payload = {
    #         "model": "sglang_model",
    #         "messages": messages,
    #         "max_tokens": 512,
    #         "temperature": 0.7,
    #     }

    #     async with aiohttp.ClientSession() as session:
    #         async with session.post(url, json=payload) as resp:
    #             data = await resp.json()
    #             return data["choices"][0]["message"]["content"]


    # async def _async_rollout_a_request(self, req, **kwargs):
    #     # your SE logic
    #     ...

# @dataclass
# class Trajectory:
#     steps: List[str] = field(default_factory=list)
#     reward: Optional[float] = None
#     fingerprint: str = ""


# class SERollout(SGLangRollout):
#     """
#     """

#     def __init__(
#         self,
#         engine_caller,      
#         tokenizer,
#         max_new_tokens: int = 512,
#         budget: int = 20,
#         max_iter: int = 4,
#         eps: float = 0.01,
#         k_elite: int = 10,
#     ):
#         self.engine = engine_caller
#         self.tokenizer = tokenizer
#         self.budget = budget
#         self.max_iter = max_iter
#         self.eps = eps
#         self.k_elite = k_elite
#         self.max_new_tokens = max_new_tokens

#     async def __call__(self, task_desc: str) -> Trajectory:
#         pool = await self._generate_initial_pool(task_desc)
#         best_reward = -math.inf
#         best_traj = None
#         api_calls = 0

#         for itr in range(self.max_iter):
#             # 1. Revision
#             revised = []
#             for τ in pool:
#                 refl = await self._reflect(τ, task_desc)
#                 api_calls += 1
#                 τ_rev = await self._revise(τ, refl)
#                 api_calls += 1
#                 revised.append(τ_rev)
#             pool += revised

#             # 2. Recombination
#             new = []
#             # Crossover
#             for _ in range(3):
#                 ta, tb = random.sample(pool, 2)
#                 new.append(self._crossover(ta, tb))
#                 api_calls += 1
#             # Transfer
#             for _ in range(3):
#                 t_weak = min(pool, key=lambda t: t.reward or -math.inf)
#                 t_strong = max(pool, key=lambda t: t.reward or -math.inf)
#                 new.append(await self._transfer(t_weak, [t_strong]))
#                 api_calls += 1
#             # Restructure
#             new.append(await self._restructure(pool))
#             api_calls += 1
#             pool += new

#             # 3. Evaluate
#             for τ in pool:
#                 if τ.reward is None:
#                     τ.reward = await self._evaluate(τ, task_desc)
#                     api_calls += 1

#             pool = self._select_elite_diverse(pool, self.k_elite)
#             max_r = max(t.reward for t in pool)
#             if max_r > best_reward:
#                 best_reward = max_r
#                 best_traj = max(pool, key=lambda t: t.reward or -math.inf)

#             if api_calls >= self.budget or (itr and max_r - best_reward < self.eps):
#                 break

#         return best_traj

#     async def _llm(self, prompt: str) -> str:
#         messages = [Message(role="user", content=prompt)]
#         resp = await self.engine(
#             {
#                 "messages": [m.model_dump() for m in messages],
#                 "max_new_tokens": self.max_new_tokens,
#                 "temperature": 0.7,
#             }
#         )
#         return resp["choices"][0]["message"]["content"]

#     async def _generate_initial_pool(self, task: str) -> List[Trajectory]:
#         pool = []
#         for plan_id in range(5):
#             τ = Trajectory(steps=(await self._llm(self._plan_prompt(task, plan_id))).splitlines())
#             pool.append(τ)
#             pool.append(Trajectory(steps=(await self._llm(self._mutate_prompt(τ, mild=True))).splitlines()))
#             pool.append(Trajectory(steps=(await self._llm(self._mutate_prompt(τ, mild=False))).splitlines()))
#         return pool[-10:]

#     async def _reflect(self, τ: Trajectory, task: str) -> str:
#         prompt = f"Trajectory:\n{chr(10).join(τ.steps)}\nTask:\n{task}\nIdentify critical step & weaknesses."
#         return await self._llm(prompt)

#     async def _revise(self, τ: Trajectory, reflection: str) -> Trajectory:
#         prompt = f"Reflection:\n{reflection}\nRevise trajectory:\n{chr(10).join(τ.steps)}"
#         new_steps = (await self._llm(prompt)).splitlines()
#         return Trajectory(steps=new_steps)

#     def _crossover(self, ta: Trajectory, tb: Trajectory) -> Trajectory:
#         cut = random.randint(1, min(len(ta.steps), len(tb.steps)) - 1)
#         new_steps = ta.steps[:cut] + tb.steps[cut:]
#         return Trajectory(steps=new_steps)

#     async def _transfer(self, target: Trajectory, refs: List[Trajectory]) -> Trajectory:
#         prompt = (
#             f"Target:\n{chr(10).join(target.steps)}\n"
#             f"Ref strengths:\n{chr(10).join(refs[0].steps)}\n"
#             f"Enhance target with insights from ref."
#         )
#         new_steps = (await self._llm(prompt)).splitlines()
#         return Trajectory(steps=new_steps)

#     async def _restructure(self, pool: List[Trajectory]) -> Trajectory:
#         prompt = "Summarize global patterns and create a concise new trajectory."
#         new_steps = (await self._llm(prompt)).splitlines()
#         return Trajectory(steps=new_steps)

#     async def _evaluate(self, τ: Trajectory, task: str) -> float:
#         task_score = 1.0  # run_unit_tests(τ)
#         qual_score = 5.0  # await self._llm_score_reasoning(τ)
#         eff_score = 1 / (len(τ.steps) + 1)
#         return 0.5 * task_score + 0.3 * qual_score + 0.2 * eff_score

#     def _select_elite_diverse(self, pool: List[Trajectory], k: int) -> List[Trajectory]:
#         pool = sorted(pool, key=lambda t: t.reward or -math.inf, reverse=True)
#         elites = pool[: k // 2]
#         rest = pool[k // 2 :]
#         diverse = []
#         for t in rest:
#             if all(self._sim(t.fingerprint, e.fingerprint) < 0.8 for e in elites + diverse):
#                 diverse.append(t)
#                 if len(diverse) >= k // 2:
#                     break
#         return elites + diverse

#     def _sim(self, a: str, b: str) -> float:
#         return 0.0

#     def _plan_prompt(self, task: str, plan_id: int) -> str:
#         return f"Generate a {plan_id}-th step-by-step plan to solve:\n{task}"

#     def _mutate_prompt(self, τ: Trajectory, mild: bool):
#         return f"{'Slightly' if mild else 'Drastically'} mutate trajectory:\n{chr(10).join(τ.steps)}"