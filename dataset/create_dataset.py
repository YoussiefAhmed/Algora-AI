import pandas as pd
import random

rows = []

for i in range(2000):
    t = random.choice(["spirulina", "chlorella"])

    if t == "spirulina":
        temp = random.uniform(24, 36)
        light = random.uniform(200, 900)
        ph = random.uniform(7.5, 10.5)
        healthy = 1 if (27<=temp<=32 and 350<=light<=800 and 8.2<=ph<=9.8) else 0

    else:
        temp = random.uniform(18, 32)
        light = random.uniform(100, 700)
        ph = random.uniform(6, 8)
        healthy = 1 if (22<=temp<=28 and 200<=light<=600 and 6.6<=ph<=7.4) else 0

    if random.random() < 0.15:
        healthy = 1 - healthy

    rows.append([temp, light, ph, t, healthy])

df = pd.DataFrame(rows, columns=["temp", "light", "ph", "type", "label"])
df.to_csv("algae_dataset.csv", index=False)
print("Dataset Created!")