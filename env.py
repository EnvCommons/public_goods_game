import asyncio
import textarena as ta
import re
import openai
from typing import List, Tuple
from pydantic import BaseModel
from openreward.environments import Environment, JSONObject, ToolOutput, TextBlock, tool


class TaskSpec(BaseModel):
    id: str
    env_id: str
    seed: int
    variant: str = ""


class SendMessageParams(BaseModel, extra="forbid"):
    message: str


class ContributeParams(BaseModel, extra="forbid"):
    tokens: int


class PublicGoodsGameEnvironment(Environment):
    GAME_NAME = "PublicGoodsGame"
    VARIANTS = [
        "PublicGoodsGame-v0",
        "PublicGoodsGame-v0-train",
        "PublicGoodsGame-v0-raw",
    ]
    NUM_TASKS_PER_VARIANT = 50
    AGENT_PLAYER_ID = 0
    NUM_PLAYERS = 3

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.secrets = secrets

        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError("openai_api_key required in secrets for PublicGoodsGame (LLM opponents)")
        self.opponent_client = openai.AsyncClient(api_key=api_key)

        self.ta_env = ta.make(env_id=self.config.env_id)
        self.game_done = False
        self.turn_count = 0
        self.opponent_histories: dict[int, list] = {}

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        tasks = []
        for variant_id in cls.VARIANTS:
            for seed_idx in range(cls.NUM_TASKS_PER_VARIANT):
                seed = seed_idx if split == "train" else seed_idx + 10000
                tasks.append({
                    "id": f"{variant_id}_seed{seed}",
                    "env_id": variant_id,
                    "seed": seed,
                    "variant": variant_id,
                })
        return tasks

    def _format_observation(self, observation) -> str:
        if isinstance(observation, str):
            match = None
            for m in re.finditer(r'^\[(?!GAME\])[^\]]+\].*$', observation, re.MULTILINE):
                match = m
            if match:
                return observation[match.end():].lstrip('\n')
            return observation
        if isinstance(observation, list):
            if not observation:
                return ""
            last = observation[-1]
            if isinstance(last, tuple) and len(last) >= 2:
                return str(last[1])
            return str(last)
        return str(observation)

    def _map_reward(self, ta_rewards: dict, player_id: int) -> float:
        raw = ta_rewards.get(player_id, 0)
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    def _opponent_system_prompt(self, player_id: int) -> str:
        return (
            f"You are Player {player_id} in a 3-player Public Goods Game.\n\n"
            f"Rules:\n"
            f"- Each round every player receives 20 tokens.\n"
            f"- During communication phases, you discuss strategy with other players.\n"
            f"- During the decision phase, you choose how many tokens (0-20) to contribute to the public pot.\n"
            f"- All contributions are multiplied by 1.5 and split equally among all players.\n"
            f"- Your payoff = tokens you kept + your share of the public pot.\n"
            f"- The game lasts 3 rounds, each with communication phases then a decision.\n\n"
            f"For chat turns, send a short public message.\n"
            f"For contribution turns, respond with [X] where X is your contribution (0-20).\n"
            f"Respond with ONLY your message or contribution decision."
        )

    async def _get_opponent_action(self, observation: str, player_id: int) -> str:
        if player_id not in self.opponent_histories:
            self.opponent_histories[player_id] = [
                {"role": "system", "content": self._opponent_system_prompt(player_id)}
            ]

        self.opponent_histories[player_id].append({"role": "user", "content": observation})

        try:
            response = await self.opponent_client.chat.completions.create(
                model="gpt-5-mini",
                messages=self.opponent_histories[player_id],
            )
            reply = response.choices[0].message.content.strip()
            self.opponent_histories[player_id].append({"role": "assistant", "content": reply})
            return reply
        except Exception:
            return "[10]"

    async def _run_opponent_turns(self, current_player_id: int, current_observation) -> str:
        while current_player_id != self.AGENT_PLAYER_ID:
            obs_text = current_observation if isinstance(current_observation, str) else str(current_observation)
            opponent_action = await self._get_opponent_action(obs_text, current_player_id)
            done, info = self.ta_env.step(action=opponent_action)
            if done:
                self.game_done = True
                return opponent_action
            current_player_id, current_observation = self.ta_env.get_observation()
        return self._format_observation(current_observation)

    async def get_prompt(self) -> List[TextBlock]:
        self.ta_env.reset(num_players=self.NUM_PLAYERS, seed=self.config.seed)
        player_id, observation = self.ta_env.get_observation()

        if player_id != self.AGENT_PLAYER_ID:
            obs_text = await self._run_opponent_turns(player_id, observation)
        else:
            obs_text = self._format_observation(observation)

        prompt = (
            f"You are Player 0 in a Public Goods Game with 3 players.\n\n"
            f"Each round you receive 20 tokens. You can communicate with others, then decide how many to contribute.\n"
            f"Payoff = tokens kept + your share of the public good (contributions multiplied by 1.5, split equally).\n\n"
            f"Use send_message for chat turns, and contribute(tokens) for contribution decisions.\n\n"
            f"{obs_text}"
        )
        return [TextBlock(text=prompt)]

    def _handle_game_end(self) -> Tuple[str, float, bool]:
        rewards, game_info = self.ta_env.close()
        reward = self._map_reward(rewards, self.AGENT_PLAYER_ID)
        reason = ""
        if isinstance(game_info, dict) and self.AGENT_PLAYER_ID in game_info:
            reason = game_info[self.AGENT_PLAYER_ID].get("reason", "")
        summary = f"Game Over! Your reward: {reward:.2f}"
        if reason:
            summary += f"\n{reason}"
        self.game_done = True
        return summary, reward, True

    async def _do_action(self, action: str) -> ToolOutput:
        if self.game_done:
            return ToolOutput(
                blocks=[TextBlock(text="Game is already over.")],
                metadata={"error": "game_finished"},
                reward=0.0,
                finished=True,
            )

        done, info = self.ta_env.step(action=action)
        self.turn_count += 1

        if done:
            summary, reward, finished = self._handle_game_end()
            return ToolOutput(
                blocks=[TextBlock(text=summary)],
                metadata={"turn": self.turn_count, "reward": reward},
                reward=reward,
                finished=True,
            )

        player_id, observation = self.ta_env.get_observation()
        if player_id != self.AGENT_PLAYER_ID:
            after_move_obs = self._format_observation(observation)
            obs_text = await self._run_opponent_turns(player_id, observation)
            if self.game_done:
                summary, reward, finished = self._handle_game_end()
                return ToolOutput(
                    blocks=[TextBlock(text=f"After your move:\n{after_move_obs}\n\nOpponent's response:\n{obs_text}\n\n{summary}")],
                    metadata={"turn": self.turn_count, "reward": reward},
                    reward=reward,
                    finished=True,
                )
            obs_text = f"After your move:\n{after_move_obs}\n\nAfter opponent's response:\n{obs_text}"
        else:
            obs_text = self._format_observation(observation)

        return ToolOutput(
            blocks=[TextBlock(text=obs_text)],
            metadata={"turn": self.turn_count},
            reward=0.0,
            finished=False,
        )

    @tool
    async def send_message(self, params: SendMessageParams) -> ToolOutput:
        """Send a chat message during the communication phase."""
        return await self._do_action(f"{{{params.message}}}")

    @tool
    async def contribute(self, params: ContributeParams) -> ToolOutput:
        """Contribute tokens to the public pot (0-20)."""
        return await self._do_action(f"[{params.tokens}]")
