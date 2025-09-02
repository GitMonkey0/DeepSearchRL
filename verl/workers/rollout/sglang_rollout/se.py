from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from verl import DataProto
from verl.workers.rollout.sglang_rollout.schemas import Message
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj


@dataclass
class Trajectory:
    steps: List[str] = field(default_factory=list)
    reward: Optional[float] = None
    fingerprint: str = ""


class SE:
    """
    """

    def __init__(
        self,
        engine_caller,      
        tokenizer,
        max_new_tokens: int = 512,
        budget: int = 20,
        max_iter: int = 4,
        eps: float = 0.01,
        k_elite: int = 10,
    ):
        self.engine = engine_caller
        self.tokenizer = tokenizer
        self.budget = budget
        self.max_iter = max_iter
        self.eps = eps
        self.k_elite = k_elite
        self.max_new_tokens = max_new_tokens

    async def __call__(self, task_desc: str) -> Trajectory:
        pool = await self._generate_initial_pool(task_desc)
        best_reward = -math.inf
        best_traj = None
        api_calls = 0

        for itr in range(self.max_iter):
            # 1. Revision
            revised = []
            for τ in pool:
                refl = await self._reflect(τ, task_desc)
                api_calls += 1
                τ_rev = await self._revise(τ, refl)
                api_calls += 1
                revised.append(τ_rev)
            pool += revised

            # 2. Recombination
            new = []
            # Crossover
            for _ in range(3):
                ta, tb = random.sample(pool, 2)
                new.append(self._crossover(ta, tb))
                api_calls += 1
            # Transfer
            for _ in range(3):
                t_weak = min(pool, key=lambda t: t.reward or -math.inf)
                t_strong = max(pool, key=lambda t: t.reward or -math.inf)
                new.append(await self._transfer(t_weak, [t_strong]))
                api_calls += 1
            # Restructure
            new.append(await self._restructure(pool))
            api_calls += 1
            pool += new

            # 3. Evaluate
            for τ in pool:
                if τ.reward is None:
                    τ.reward = await self._evaluate(τ, task_desc)
                    api_calls += 1

            pool = self._select_elite_diverse(pool, self.k_elite)
            max_r = max(t.reward for t in pool)
            if max_r > best_reward:
                best_reward = max_r
                best_traj = max(pool, key=lambda t: t.reward or -math.inf)

            if api_calls >= self.budget or (itr and max_r - best_reward < self.eps):
                break

        return best_traj

    async def _llm(self, prompt: str) -> str:
        messages = [Message(role="user", content=prompt)]
        resp = await self.engine(
            {
                "messages": [m.model_dump() for m in messages],
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.7,
            }
        )
        return resp["choices"][0]["message"]["content"]

    async def _generate_initial_pool(self, task: str) -> List[Trajectory]:
        pool = []
        for plan_id in range(5):
            τ = Trajectory(steps=(await self._llm(self._plan_prompt(task, plan_id))).splitlines())
            pool.append(τ)
            pool.append(Trajectory(steps=(await self._llm(self._mutate_prompt(τ, mild=True))).splitlines()))
            pool.append(Trajectory(steps=(await self._llm(self._mutate_prompt(τ, mild=False))).splitlines()))
        return pool[-10:]

    async def _reflect(self, τ: Trajectory, task: str) -> str:
        prompt = f"Trajectory:\n{chr(10).join(τ.steps)}\nTask:\n{task}\nIdentify critical step & weaknesses."
        return await self._llm(prompt)

    async def _revise(self, τ: Trajectory, reflection: str) -> Trajectory:
        prompt = f"Reflection:\n{reflection}\nRevise trajectory:\n{chr(10).join(τ.steps)}"
        new_steps = (await self._llm(prompt)).splitlines()
        return Trajectory(steps=new_steps)

    def _crossover(self, ta: Trajectory, tb: Trajectory) -> Trajectory:
        cut = random.randint(1, min(len(ta.steps), len(tb.steps)) - 1)
        new_steps = ta.steps[:cut] + tb.steps[cut:]
        return Trajectory(steps=new_steps)

    async def _transfer(self, target: Trajectory, refs: List[Trajectory]) -> Trajectory:
        prompt = (
            f"Target:\n{chr(10).join(target.steps)}\n"
            f"Ref strengths:\n{chr(10).join(refs[0].steps)}\n"
            f"Enhance target with insights from ref."
        )
        new_steps = (await self._llm(prompt)).splitlines()
        return Trajectory(steps=new_steps)

    async def _restructure(self, pool: List[Trajectory]) -> Trajectory:
        prompt = "Summarize global patterns and create a concise new trajectory."
        new_steps = (await self._llm(prompt)).splitlines()
        return Trajectory(steps=new_steps)

    async def _evaluate(self, τ: Trajectory, task: str) -> float:
        task_score = 1.0  # run_unit_tests(τ)
        qual_score = 5.0  # await self._llm_score_reasoning(τ)
        eff_score = 1 / (len(τ.steps) + 1)
        return 0.5 * task_score + 0.3 * qual_score + 0.2 * eff_score

    def _select_elite_diverse(self, pool: List[Trajectory], k: int) -> List[Trajectory]:
        pool = sorted(pool, key=lambda t: t.reward or -math.inf, reverse=True)
        elites = pool[: k // 2]
        rest = pool[k // 2 :]
        diverse = []
        for t in rest:
            if all(self._sim(t.fingerprint, e.fingerprint) < 0.8 for e in elites + diverse):
                diverse.append(t)
                if len(diverse) >= k // 2:
                    break
        return elites + diverse

    def _sim(self, a: str, b: str) -> float:
        return 0.0

    def _plan_prompt(self, task: str, plan_id: int) -> str:
        return f"Generate a {plan_id}-th step-by-step plan to solve:\n{task}"

    def _mutate_prompt(self, τ: Trajectory, mild: bool):
        return f"{'Slightly' if mild else 'Drastically'} mutate trajectory:\n{chr(10).join(τ.steps)}"