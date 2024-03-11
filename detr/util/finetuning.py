import torch
import src.main as mn


if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mn.main('train', 'detection', './table-transformer/src/detection_config.json', './dataset/')
