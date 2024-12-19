import argparse
from data import get_mnist_sample
from tsne import TSNE


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--sample_size', type=int, help="mnist sample-size", default=1000
    )
    parser.add_argument("--perplexity", type=int, help="which find sigma", default=30)
    parser.add_argument(
        "--learning_rate", type=float, help="step size of iteration", default=200.0
    )
    parser.add_argument("--max_iter", type=int, help="training iteration", default=2000)
    parser.add_argument("--early_exaggeration", type=int, help="", default=4)

    args, _ = parser.parse_known_args()

    return args


def main():
    args = get_args()

    x, y = get_mnist_sample(args.sample_size)
    t_sne = TSNE()

    y_tsne = t_sne.train(
        x, args.perplexity, args.learning_rate, args.max_iter, args.early_exaggeration
    )
    t_sne.plot_tsne(y_tsne, y)


if __name__ == "__main__":
    main()
