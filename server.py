from openreward.environments import Server

from safetyclassify import SafetyClassify

if __name__ == "__main__":
    server = Server([SafetyClassify])
    server.run()
