import matplotlib.pyplot as plt

def plot_fashion_items(images, labels, n_cols=5):
    """
    Función para visualizar un subset de instancias con su etiqueta
    """
    # Etiquetas de Fashion MNIST
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    label_names = [class_names[label] for label in labels]
    n_rows = (len(images) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols * 1.5, n_rows * 1.7))
    for index, (image, label) in enumerate(zip(images, label_names)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()