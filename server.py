from openreward.environments import Server
from env import PublicGoodsGameEnvironment

if __name__ == "__main__":
    server = Server([PublicGoodsGameEnvironment])
    server.run()
