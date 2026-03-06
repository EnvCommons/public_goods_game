# PublicGoodsGame

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/PublicGoodsGame)

## Description

**PublicGoodsGame** is an environment for evaluating agents on economic decision-making and social cooperation in a public goods game. This environment wraps the PublicGoodsGame implementation from [TextArena](https://github.com/LeonGuertler/TextArena), a framework for text-based game environments.

## Capabilities

- Economic reasoning and resource allocation
- Strategic communication and persuasion
- Trust and cooperation building
- Balancing individual and collective incentives

## Compute Requirements

PublicGoodsGame does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://github.com/LeonGuertler/TextArena/blob/main/LICENSE).

## Tasks

There are two splits: train (150 tasks) and test (150 tasks). Each split contains 50 tasks across each of 3 variants:

- **PublicGoodsGame-v0**
- **PublicGoodsGame-v0-train**
- **PublicGoodsGame-v0-raw**

Each task is seeded for reproducibility.

## Reward Structure

This is a sparse reward environment. Rewards are mapped from TextArena's native range of {-1, 0, 1} to {0.0, 0.5, 1.0} via `(raw + 1) / 2`.

We do not use LLM graders for this environment; reward is determined programmatically.

## Data

Game state is generated procedurally by the TextArena engine using seeded randomness. No external data files are required.

## Tools

Agents are given two tools:

- `send_message(message)`: Send a chat message during the communication phase.
- `contribute(tokens)`: Contribute tokens to the public pot (0-20).

## Time Horizon

PublicGoodsGame is a multi-turn environment.

## Environment Difficulty

Medium. Agents must navigate the tension between maximizing personal payoff and contributing to the collective good, while communicating with other players to coordinate contributions.

## Other Environment Requirements

This environment requires an OpenAI API key (passed via secrets) to power the LLM opponents.

## Safety

Agents in PublicGoodsGame interact only with an economic simulation and have no access to external systems, the internet, or sensitive data. The environment does not present safety risks.

## Citations

```bibtex
@software{textarena2024,
  author    = {Guertler, Leon and Banting, Wilfried and Pignatelli, Eduardo},
  title     = {TextArena},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/LeonGuertler/TextArena}
}
```
