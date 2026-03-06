import asyncio
import textarena as ta
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
            return observation
        if isinstance(observation, list):
            parts = []
            for item in observation:
                if isinstance(item, tuple) and len(item) >= 2:
                    parts.append(str(item[1]))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(observation)

    def _map_reward(self, ta_rewards: dict, player_id: int) -> float:
        raw = ta_rewards.get(player_id, 0)
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    async def _get_opponent_action(self, observation: str, player_id: int) -> str:
        system_prompt = (
            f"You are Player {player_id} in a Public Goods Game. "
            f"For chat turns, send a short message. "
            f"For contribution turns, respond with [contribute X] where X is 0-20. "
            f"Respond with ONLY your message or contribution."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
        ]
        try:
            response = await self.opponent_client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "[contribute 10]"

    async def _run_opponent_turns(self, current_player_id: int, current_observation) -> str:
        while current_player_id != self.AGENT_PLAYER_ID:
            obs_text = self._format_observation(current_observation)
            opponent_action = await self._get_opponent_action(obs_text, current_player_id)
            done, info = self.ta_env.step(action=opponent_action)
            if done:
                self.game_done = True
                return ""
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
            obs_text = await self._run_opponent_turns(player_id, observation)
            if self.game_done:
                summary, reward, finished = self._handle_game_end()
                return ToolOutput(
                    blocks=[TextBlock(text=summary)],
                    metadata={"turn": self.turn_count, "reward": reward},
                    reward=reward,
                    finished=True,
                )
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
        return await self._do_action(params.message)

    @tool
    async def contribute(self, params: ContributeParams) -> ToolOutput:
        """Contribute tokens to the public pot (0-20)."""
        return await self._do_action(f"[contribute {params.tokens}]")
