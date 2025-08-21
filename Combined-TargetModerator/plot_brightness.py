import numpy as np
import matplotlib.pyplot as plt

data = np.array([
[0, 4, np.float64(0.001575755108362201), np.float64(0.00034760157633291754)],
[2, 4, np.float64(0.0019252911965590574), np.float64(0.00020158141471447277)],
[4, 4, np.float64(0.0015803143365522494), np.float64(0.00015702658745104728)],
[6, 4, np.float64(0.0009113123963991675), np.float64(9.093309633731503e-05)],
[8, 4, np.float64(0.0007169334643071177), np.float64(0.0003172298421610429)],
#[4, 0, np.float64(0.00245250177059078), np.float64(0.00040268018143780227)],
#[4, 2, np.float64(0.0016303274845801438), np.float64(0.00017958957049845871)],
#[4, 4, np.float64(0.0015803143365522494), np.float64(0.00015702658745104728)],
#[4, 6, np.float64(0.001068826849337948), np.float64(0.00010053454491308976)],
#[4, 8, np.float64(0.0010277080976264738), np.float64(0.0001991243673433931)],
])

labels = [f"{int(x)}x{int(y)}" for x, y in data[:, :2]]
values = data[:, 2]
errors = data[:, 3]

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(labels, values, yerr=errors, fmt='o', capsize=5)
plt.xlabel("Plane Count Along z")
plt.ylabel("Brightness")
plt.title("Brightness vs. Plane Count")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
