import torch.cuda


def main():
    if not torch.cuda.is_available():
        print(f'This code need CUDA')
        return


if __name__ == '__main__':
    main()
