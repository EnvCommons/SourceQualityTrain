from openreward.environments import Server

from sourcequalitytrain import SourceQualityTrain

if __name__ == "__main__":
    server = Server([SourceQualityTrain])
    server.run()
