import numpy as np
import matplotlib.pyplot as plt

total_epochs = 10000
initial_lr = 0.1
# decay_rates = [0.96, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999]
decay_rates = [ 0.997, 0.998, 0.999]
epochs = np.arange(total_epochs//2, total_epochs)
for rate in decay_rates:
    scale = 1000
    lrs = initial_lr * (rate ** ((epochs - 500)/scale))
    plt.plot(epochs, lrs, label=f"rate={rate}")
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.title("Learning Rate Decay Comparison")
plt.savefig("exp_decay_rate_comparison1.png")
