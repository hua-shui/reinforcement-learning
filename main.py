def main():
    import torch
    print("显卡是否可用:", torch.cuda.is_available())
    print("显卡名字:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")


if __name__ == "__main__":
    main()
