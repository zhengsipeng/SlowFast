import os

def download_ava():
    with open("ava_file_names_trainval_v2.1.txt", "r") as f:
        ls = [l.strip() for l in f.readlines()]
    if not os.path.exists("trainval"):
        os.makedirs("trainval")
    
    if not os.path.exists("test"):
        os.makedirs("test")

if __name__ == "__main__":
    download_ava()