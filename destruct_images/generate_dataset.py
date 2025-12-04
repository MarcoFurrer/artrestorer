from overlay_transparent import main
import os

if __name__ == "__main__":
    for file in os.listdir("train_1"):
        main(
            os.path.join("train_1", file),
            os.path.join("train_1_destructed", file),
            os.path.join("train_1_mask", file),
            save_type="JPEG",
        )