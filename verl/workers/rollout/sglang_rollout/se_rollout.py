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
from verl.utils.reward_score.search_r1_like_qa_em import compute_score
from collections import defaultdict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


REVISION_SYSTEM = r'''
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


CROSSOVER_SYSTEM = r"""You are an expert in analyzing and synthesizing agent trajectories. \
Your task is to critically analyze two different trajectories and create a new optimized trajectory \
by combining the best elements from both approaches.

Your fusion process should:
1. Identify the strengths and weaknesses of each trajectory
2. Extract the most effective strategies and techniques from both
3. Creatively integrate these elements into a coherent new trajectory
4. Ensure the new trajectory maintains logical flow and consistency
5. Avoid simply concatenating the trajectories—create genuine synthesis

You need to analyze both trajectories and provide:
- Analysis of strengths from trajectory A
- Analysis of strengths from trajectory B
- A new fused trajectory that combines the best aspects
- Rationale for the fusion decisions

Output must strictly follow JSON format, containing:
- trajectory_a_strengths: list of strengths from first trajectory
- trajectory_b_strengths: list of strengths from second trajectory
- fused_trajectory: new trajectory steps combining best elements
- fusion_rationale: explanation of fusion decisions
"""

TRANSFER_SYSTEM = r"""You are an expert in optimizing agent trajectories through transfer learning. \
Your task is to enhance a target trajectory by transferring effective strategies, insights, \
and approaches from a pool of reference trajectories.

Your transfer learning process should:
1. Analyze the target trajectory to identify areas for improvement
2. Extract valuable patterns, strategies and techniques from the reference trajectories
3. Transfer these elements to enhance the target trajectory
4. Ensure the enhanced trajectory maintains logical coherence and consistency
5. Focus on meaningful knowledge transfer, not simply adding steps

Target Trajectory: {trajectory_target}

Reference Trajectory Pool: {traj_pool}

Carefully analyze both the target trajectory and reference pool, then create an enhanced \
version of the target trajectory that incorporates the most valuable elements from the \
reference trajectories.

Your output should be a single JSON object representing the enhanced trajectory, \
following this exact format:
{
  "trajectory": [
    {"step": 1, "action": "...", "observation": "..."},
    {"step": 2, "action": "...", "observation": "..."},
    ...
  ]
}

Make sure the enhanced trajectory:
- Addresses weaknesses in the original target trajectory
- Incorporates valuable insights from reference trajectories
- Maintains a coherent problem-solving approach
- Includes specific implementation details
- Has logical progression between steps
- Is complete enough to solve the task effectively
"""

RESTRUCTURE_SYSTEM = r"""You are an expert in large-scale trajectory restructuring. \
Your task is to synthesize a new reasoning trajectory by analyzing the global structure \
of a trajectory population. Unlike Crossover or Transfer that focus on local segment \
manipulation, this task requires holistic restructuring based on global insights across \
all input trajectories.

Your restructuring process should:
1. Analyze the entire trajectory pool to discover abstract patterns, common subgoals, \
   and shared structures
2. Identify redundant reasoning paths and filter out ineffective or repetitive steps
3. Synthesize a completely new trajectory that aligns with the overall problem-solving \
   objective, but reflects a novel and optimized reasoning process
4. Maintain logical consistency, completeness, and step-wise progression of the trajectory

Trajectory Pool: {trajectory_pool}

You must generate a single restructured trajectory that combines the collective strengths \
and high-level reasoning strategies inferred from the input trajectories. Your output \
should be a single JSON object representing the newly restructured trajectory, \
following this exact format:
{
  "new_trajectory": [
    {
      "step": 1,
      "action": "action description",
      "observation": "observation description",
      "reasoning": "why this step was chosen and how it contributes"
    },
    ...
  ]
}

Make sure the restructured trajectory:
- Reflects global insights derived from the trajectory pool
- Avoids redundancy and overly local reasoning
- Introduces a coherent and efficient solution strategy
- Demonstrates abstract synthesis and long-range planning
- Forms a complete and executable path to solve the task
"""

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

        for turn in dialog:
            if turn.get("role") == "user":
                text = turn.get("content", "").strip()
                if text:
                    trace.append({"step": step, "type": "user", "content": text})
                    step += 1
                break   # 只要第一条 user

        for turn in dialog:
            role = turn.get("role")
            text = turn.get("content", "")
            tool_calls = turn.get("tool_calls")

            if role in {"system", "user"}:
                continue

            if role == "assistant":
                if text.strip():
                    trace.append({"step": step, "type": "think", "content": text.strip()})
                    step += 1

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

                m = re.search(r"<answer>(.*?)</answer>", text, re.S)
                if m:
                    trace.append({"step": step, "type": "answer", "content": m.group(1).strip()})
                    step += 1

            elif role == "tool":
                trace.append({"step": step, "type": "information", "content": text.strip()})
                step += 1

        return trace

    def _extract_critical_step_json(self, text: str):
        """
        从LLM响应中提取JSON内容
        
        参数:
            response_text (str): 可能包含JSON和自然语言的文本
        
        返回:
            dict: 解析出的JSON字典，如果解析失败则返回None
        """
        json_pattern = r'json\s({.?})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if not match:
            json_pattern = r'({.*})'
            match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                try:
                    return json.loads(match.group(1).replace("'", '"'))
                except:
                    return None
        return None

    def _revision(self, prompts, init_traj) -> DataProto:
        traj_formatted = []
        for m in init_traj.non_tensor_batch.get("messages", []):
            messages = [message.model_dump() for message in m["messages"]]
            traj_formatted.append(self._parse_conversation(messages))

        data = [[
            {"role": "system", "content": REVISION_SYSTEM},
            {"role": "user", "content": str(traj)}  
        ] for traj in traj_formatted]
        
        input_ids = torch.tensor([0] * len(prompts.batch["input_ids"]), device=prompts.batch["input_ids"].device)  
        inputs = DataProto.from_dict(
            tensors = {
                "input_ids": input_ids
            },
            non_tensors = {
                "raw_prompt": np.array(data, dtype=object),
                "tools_kwargs": np.array([{}] * len(data), dtype=object)
            }
        )
        outputs = self._req_level_generate_sequences(inputs)
        outputs_str = self.processing_class.batch_decode(outputs.batch["responses"], skip_special_tokens=True)
        critics = [self._extract_critical_step_json(string) for string in outputs_str]
        
        orinial_messages = prompts.non_tensor_batch.get("raw_prompt", []).tolist()
        reconstructed_messages = []
        for messages, traj, critic in zip(orinial_messages, traj_formatted, critics):
            new_messages = [
                {
                    "role": "system",
                    "content": f'''{messages[0]["content"]}\n\n
                    - Your previous trajectory: {traj}
                    - Instructor's refinement: {critic}
                    Blend the refinement into your prior trajectory and produce a fresh, revised answer; tools remain available.  
                    Write as if this were your first attempt—give no hint of external guidance.\n'''
                },
                messages[-1]
                ]
            reconstructed_messages.append(new_messages)

        prompts.non_tensor_batch["raw_prompt"] = np.array(reconstructed_messages, dtype=object)
        
        revised_traj = self._req_level_generate_sequences(prompts)

        return revised_traj

    def _recombine(self, prompts, revised_traj):
        revised_str = self.processing_class.batch_decode(
            revised_traj.batch["responses"], skip_special_tokens=True
        )
        answers = prompts.non_tensor_batch.get("golden_answers").tolist()
        scores = [
            compute_score(pred, {"target": ans})["acc"]
            for pred, ans in zip(revised_str, answers)
        ]

        traj_formatted = []
        for m in revised_traj.non_tensor_batch.get("messages", []):
            messages = [message.model_dump() for message in m["messages"]]
            traj_formatted.append(str(self._parse_conversation(messages)))
        index_arr = prompts.non_tensor_batch["index"].tolist()
        
        group = defaultdict(list)          # index -> [(score, traj_str, global_pos), ...]
        for pos, (idx, sc, tr) in enumerate(zip(index_arr, scores, traj_formatted)):
            group[idx].append((sc, tr, pos))
        breakpoint()
        pairs = []          # [(global_pos_A, global_pos_B, trajA, trajB), ...]
        pos2pair = {}       # global_pos -> 在 pairs 里的序号
        for idx, members in group.items():
            members.sort(key=lambda x: x[0], reverse=True)
            for i in range(0, len(members) - 1, 2):
                p1, p2 = members[i][2], members[i + 1][2]
                pairs.append((p1, p2, members[i][1], members[i + 1][1]))
                pos2pair[p1] = len(pairs) - 1
                pos2pair[p2] = len(pairs) - 1
            if len(members) % 2:
                pos2pair[members[-1][2]] = None   # 标记“无需交叉”

        if not pairs:      # 无配对直接返回
            return revised_traj
        batch_msgs = [
            [
                {"role": "system", "content": CROSSOVER_SYSTEM},
                {"role": "user", "content": f"Trajectory A: {trajA}\n\nTrajectory B: {trajB}"}
            ]
            for _, _, trajA, trajB in pairs
        ]
        device = revised_traj.batch["input_ids"].device
        inputs = DataProto.from_dict(
            tensors={"input_ids": torch.zeros(len(batch_msgs), 1, device=device)},
            non_tensors={
                "raw_prompt": np.array(batch_msgs, dtype=object),
                "tools_kwargs": np.array([{}] * len(batch_msgs), dtype=object)
            }
        )
        outputs = self._req_level_generate_sequences(inputs)
        fused_strs = self.processing_class.batch_decode(
            outputs.batch["responses"], skip_special_tokens=True
        )

        new_traj_map = {}                       # global_pos -> 新轨迹字符串
        for pair_idx, raw in enumerate(fused_strs):
            fused_json = self._extract_critical_step_json(raw) or {}
            new_traj_str = json.dumps(
                fused_json.get("fused_trajectory", []), ensure_ascii=False
            )
            pos_A, pos_B, _, _ = pairs[pair_idx]
            new_traj_map[pos_A] = new_traj_str
            new_traj_map[pos_B] = new_traj_str

        final_ordered = []
        for i in range(len(traj_formatted)):
            if i in new_traj_map:
                final_ordered.append(new_traj_map[i])
            else:                                   # 奇数末尾原样
                final_ordered.append(traj_formatted[i])

        revised_traj.non_tensor_batch["fused_traj_str"] = np.array(final_ordered, dtype=object)
        return revised_traj

    def _refine():
        pass

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        SE-Agent 完整流水线：
        1. 初始轨迹  → 2. Revision  → 3. Crossover
         → 4. Transfer → 5. Restructure → 6. 返回最优轨迹
        每一步都是**一次性批量**调用 LLM，保证效率。
        """
        init_traj = self._req_level_generate_sequences(prompts, **kwargs)

        revised_traj = self._revision(prompts, init_traj)

        cross_traj = self._recombine(prompts, revised_traj)

        transfer_traj = self._transfer(prompts, cross_traj)

        restruct_traj = self._restructure(prompts, transfer_traj)

        refine_traj = self._refine(prompts, restruct_traj)

        return refine_traj
    
    def _extract_traj_from_text(self, text: str):
        """优先提取 fused/new_traj，失败就整个 JSON 块，再失败就空"""
        try:
            root = self._extract_critical_step_json(text) or {}
            if "fused_trajectory" in root:
                return root["fused_trajectory"]
            if "trajectory" in root:
                return root["trajectory"]
            if "new_trajectory" in root:
                return root["new_trajectory"]
            if isinstance(root, list):
                return root
        except Exception:
            pass
        return []

    def _transfer(self, prompts, traj_input: DataProto) -> DataProto:
        """
        同一 index 内部：
          1. 选得分最高的一条作为 target；
          2. 其余作为 reference pool；
          3. 构造 transfer prompt 批量生成增强轨迹；
          4. 所有位置都用同一条增强结果写回。
        """
        strs = self.processing_class.batch_decode(
            traj_input.batch["responses"], skip_special_tokens=True
        )
        ans = prompts.non_tensor_batch.get("golden_answers").tolist()
        scores = [compute_score(p, {"target": a})["acc"] for p, a in zip(strs, ans)]

        trajs = []
        for m in traj_input.non_tensor_batch.get("messages", []):
            msgs = [msg.model_dump() for msg in m["messages"]]
            trajs.append(str(self._parse_conversation(msgs)))
        idx_arr = prompts.non_tensor_batch["index"].tolist()

        group = defaultdict(list)
        for pos, (idx, sc, tr) in enumerate(zip(idx_arr, scores, trajs)):
            group[idx].append((sc, tr, pos))
        for idx in group:
            group[idx].sort(key=lambda x: x[0], reverse=True)

        batch_msgs, pos2batch = [], {}  # pos -> batch_idx
        for idx, members in group.items():
            if not members:
                continue
            target_traj = members[0][1]
            pool_trajs = [m[1] for m in members[1:]]
            prompt = TRANSFER_SYSTEM.format(
                trajectory_target=target_traj,
                traj_pool=json.dumps(pool_trajs, ensure_ascii=False)
            )
            batch_msgs.append([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the enhanced trajectory."}
            ])
            for _, _, pos in members:
                pos2batch[pos] = len(batch_msgs) - 1

        if not batch_msgs:  
            return traj_input

        device = traj_input.batch["input_ids"].device
        inputs = DataProto.from_dict(
            tensors={"input_ids": torch.zeros(len(batch_msgs), 1, device=device)},
            non_tensors={
                "raw_prompt": np.array(batch_msgs, dtype=object),
                "tools_kwargs": np.array([{}] * len(batch_msgs), dtype=object)
            }
        )
        outputs = self._req_level_generate_sequences(inputs)
        out_strs = self.processing_class.batch_decode(
            outputs.batch["responses"], skip_special_tokens=True
        )

        new_map = {}
        for bidx, raw in enumerate(out_strs):
            traj_list = self._extract_traj_from_text(raw)
            new_map[bidx] = json.dumps(traj_list, ensure_ascii=False)

        final_ordered = []
        for i in range(len(trajs)):
            if i in pos2batch:
                final_ordered.append(new_map[pos2batch[i]])
            else:
                final_ordered.append(trajs[i])

        traj_input.non_tensor_batch["transfer_traj_str"] = np.array(final_ordered, dtype=object)
        return traj_input

    def _restructure(self, prompts, traj_input: DataProto) -> DataProto:
        """
        同一 index 内部：
          1. 把组内所有轨迹当 pool；
          2. 构造 restructure prompt 批量生成一条全新轨迹；
          3. 组内所有位置都用同一条新轨迹写回。
        """
        strs = self.processing_class.batch_decode(
            traj_input.batch["responses"], skip_special_tokens=True
        )
        trajs = []
        for m in traj_input.non_tensor_batch.get("messages", []):
            msgs = [msg.model_dump() for msg in m["messages"]]
            trajs.append(str(self._parse_conversation(msgs)))
        idx_arr = prompts.non_tensor_batch["index"].tolist()

        group = defaultdict(list)
        for pos, (idx, tr) in enumerate(zip(idx_arr, trajs)):
            group[idx].append((tr, pos))

        batch_msgs, pos2batch = [], {}
        for idx, members in group.items():
            pool = [m[0] for m in members]
            prompt = RESTRUCTURE_SYSTEM.format(
                trajectory_pool=json.dumps(pool, ensure_ascii=False)
            )
            batch_msgs.append([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the globally restructured trajectory."}
            ])
            for _, pos in members:
                pos2batch[pos] = len(batch_msgs) - 1

        if not batch_msgs:
            return traj_input

        device = traj_input.batch["input_ids"].device
        inputs = DataProto.from_dict(
            tensors={"input_ids": torch.zeros(len(batch_msgs), 1, device=device)},
            non_tensors={
                "raw_prompt": np.array(batch_msgs, dtype=object),
                "tools_kwargs": np.array([{}] * len(batch_msgs), dtype=object)
            }
        )
        outputs = self._req_level_generate_sequences(inputs)
        out_strs = self.processing_class.batch_decode(
            outputs.batch["responses"], skip_special_tokens=True
        )

        new_map = {}
        for bidx, raw in enumerate(out_strs):
            traj_list = self._extract_traj_from_text(raw)
            new_map[bidx] = json.dumps(traj_list, ensure_ascii=False)

        final_ordered = []
        for i in range(len(trajs)):
            if i in pos2batch:
                final_ordered.append(new_map[pos2batch[i]])
            else:
                final_ordered.append(trajs[i])

        traj_input.non_tensor_batch["restructure_traj_str"] = np.array(final_ordered, dtype=object)
        return traj_input