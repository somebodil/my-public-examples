from filelock import Timeout, FileLock

lock = FileLock("high_ground.txt.lock")
with lock:
    with open("high_ground.txt", "a") as f:
        f.write("You were the chosen one.")