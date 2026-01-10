import torch

def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())      # должно быть True
    print("CUDA device name:", torch.cuda.get_device_name(0))  


if __name__ == "__main__":
    main()