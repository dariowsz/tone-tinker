# %%
import start  # noqa isort:skip

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
data_df = pd.read_csv("data/train.csv")
wavetable_counts = data_df["osc1_wavetable"].value_counts()

# Create bar plot
plt.figure(figsize=(10, 6))
wavetable_counts.plot(kind="bar")
plt.title("Distribution of Wavetable Types")
plt.xlabel("Wavetable Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()

# %%
# Print counts and percentages
print("\nWavetable type distribution:")
print("-" * 50)
for wavetable, count in wavetable_counts.items():
    percentage = (count / len(data_df)) * 100
    print(f"{wavetable}: {count} samples ({percentage:.1f}%)")

# %%
