import matplotlib.pyplot as plt


def plot_tsne(self, y, labels, filename="tsne_result.jpg"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(y[:, 0], y[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    plt.colorbar(scatter, label="Label")
    plt.title("t-SNE Result", fontsize=14)
    plt.axis('off')
    plt.savefig(filename, dpi=300)
    plt.close()
