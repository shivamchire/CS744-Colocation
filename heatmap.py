import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv("CS744-Readings - inf-inf heatmap.csv", sep=r'\s*,\s*', engine='python')

# List of model names
models = df['Model 1'].unique()

# Create a DataFrame for the tuple strings and the minimum values
df_tuples = pd.DataFrame(index=models, columns=models)
df_min_values = pd.DataFrame(index=models, columns=models)

# Fill the DataFrame with tuple strings and minimum values from the CSV file
for _, row in df.iterrows():
    model1 = row["Model 1"]
    model2 = row["Model 2"]
    tput1 = format(row["Normalized tput 1"], '.3f')
    tput2 = format(row["Normalized tput 2"], '.3f')
    
    # Check if any of the data is None or NaN
    if pd.isnull(model1) or pd.isnull(model2) or pd.isnull(tput1) or pd.isnull(tput2):
        continue  # Skip to the next iteration

    df_tuples.loc[model1, model2] = f"({tput1}, {tput2})"
    df_min_values.loc[model1, model2] = min(float(tput1), float(tput2))

# Convert the minimum values to numbers
df_min_values = df_min_values.astype(float)
print(df_min_values)


# Create a mask for the lower triangle excluding the diagonal
mask = np.tril(np.ones_like(df_min_values, dtype=bool), k=-1)

# Plot the heatmap with the 'Paired' colormap and square cells
plt.figure(figsize=(12, 10))
sns.heatmap(df_min_values, mask=mask, annot=False, cmap='coolwarm', linewidths=5, cbar=False, square=True)

# Set the background color to light grey
plt.gca().set_facecolor('lightgrey')


# Overlay with the tuple strings only on unmasked squares
for i in range(len(df_tuples.columns)):
    for j in range(len(df_tuples)):
        if not mask[j, i]:
            plt.text(i+0.5, j+0.5, df_tuples.iloc[j, i], 
                     horizontalalignment='center', 
                     verticalalignment='center', 
                     fontsize=12)

# Move the y-axis labels to the top
plt.gca().xaxis.tick_top()

# Tilt the tick names on each axis by 30 degrees
plt.xticks(rotation=-15)
plt.yticks(rotation=-15)

# Remove the tick marks on both axes
plt.tick_params(axis='both', which='both', length=0)

plt.title("Normalized Throughput Matrix - Inference Colocation", fontsize=20, y=1.08)
plt.savefig("inf-inf heatmap.pdf")

plt.show()