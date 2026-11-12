"""A small fully connected network with Adam, in plain NumPy."""

import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for inputs, outputs in zip(sizes, sizes[1:]):
            scale = np.sqrt(2.0 / inputs)
            self.weights.append(
                (rng.standard_normal((inputs, outputs)) * scale).astype(np.float32))
            self.biases.append(np.zeros(outputs, dtype=np.float32))

    def __call__(self, x):
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(x @ w + b, 0.0)
        return x @ self.weights[-1] + self.biases[-1]

    def parameters(self):
        return self.weights + self.biases

    def save(self, path):
        arrays = {f'w{i}': w for i, w in enumerate(self.weights)}
        arrays.update({f'b{i}': b for i, b in enumerate(self.biases)})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        model = cls.__new__(cls)
        with np.load(path) as data:
            count = sum(1 for k in data.files if k.startswith('w'))
            model.weights = [data[f'w{i}'] for i in range(count)]
            model.biases = [data[f'b{i}'] for i in range(count)]
        return model

    def _gradients(self, x, output_gradient_of):
        """Backpropagate; ``output_gradient_of`` gives (loss, dLoss/dOutput)."""
        activations = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(x @ w + b, 0.0)
            activations.append(x)
        output = x @ self.weights[-1] + self.biases[-1]
        loss, grad = output_gradient_of(output)

        weight_grads = [None] * len(self.weights)
        bias_grads = [None] * len(self.biases)
        count = len(output)
        for i in range(len(self.weights) - 1, -1, -1):
            weight_grads[i] = activations[i].T @ grad / count
            bias_grads[i] = grad.mean(axis=0)
            if i:
                grad = (grad @ self.weights[i].T) * (activations[i] > 0)
        return loss, weight_grads + bias_grads


def softmax_loss(limits):
    """Cross-entropy with outputs above each row's limit masked out."""
    def bind(labels, batch):
        def loss(logits):
            mask = np.arange(logits.shape[1]) > limits[batch][:, None]
            logits = logits - logits.max(axis=1, keepdims=True)
            logits[mask] = -1e4
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            rows = np.arange(len(batch))
            value = -np.log(probs[rows, labels[batch]] + 1e-9).mean()
            grad = probs.copy()
            grad[rows, labels[batch]] -= 1.0
            grad[mask] = 0.0
            return value, grad
        return loss
    return bind


def soft_softmax_loss(limits):
    """Cross-entropy against target distributions (rows of ``labels``), with
    outputs above each row's limit masked out."""
    def bind(labels, batch):
        def loss(logits):
            mask = np.arange(logits.shape[1]) > limits[batch][:, None]
            logits = logits - logits.max(axis=1, keepdims=True)
            logits[mask] = -1e4
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            target = labels[batch]
            value = -(target * np.log(probs + 1e-9)).sum(axis=1).mean()
            grad = probs - target
            grad[mask] = 0.0
            return value, grad.astype(np.float32)
        return loss
    return bind


def huber_loss():
    def bind(labels, batch):
        def loss(outputs):
            error = outputs[:, 0] - labels[batch]
            small = np.abs(error) <= 1.0
            value = np.where(small, 0.5 * error ** 2, np.abs(error) - 0.5).mean()
            grad = np.where(small, error, np.sign(error))[:, None].astype(np.float32)
            return value, grad
        return loss
    return bind


def fit(model, inputs, labels, bind_loss, epochs=10, batch_size=4096, rate=1e-3,
        seed=0, log=print):
    """Train with Adam, keeping the weights with the best held-out loss."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(inputs))
    held_out = order[:len(inputs) // 20]
    training = order[len(inputs) // 20:]

    params = model.parameters()
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    step = 0
    best_loss = float('inf')
    best = [p.copy() for p in params]

    def evaluate():
        total = 0.0
        pieces = np.array_split(held_out, max(1, len(held_out) // batch_size))
        for piece in pieces:
            total += bind_loss(labels, piece)(model(inputs[piece]))[0]
        return total / len(pieces)

    for epoch in range(epochs):
        current = rate * 0.5 * (1.0 + np.cos(np.pi * epoch / epochs))
        shuffled = rng.permutation(training)
        for start in range(0, len(shuffled) - batch_size + 1, batch_size):
            batch = shuffled[start:start + batch_size]
            _, grads = model._gradients(inputs[batch], bind_loss(labels, batch))
            step += 1
            bias1 = 1.0 - 0.9 ** step
            bias2 = 1.0 - 0.999 ** step
            for p, g, m, v in zip(params, grads, first, second):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                p -= current * (m / bias1) / (np.sqrt(v / bias2) + 1e-8)
        loss = evaluate()
        log(f"  epoch {epoch + 1}/{epochs}: held-out loss {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best = [p.copy() for p in params]

    for p, saved in zip(params, best):
        p[...] = saved
    return best_loss
