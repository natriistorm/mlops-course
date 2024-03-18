import fire
from keywords_extraction.infer import infer
from keywords_extraction.train import train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer
        }
    )