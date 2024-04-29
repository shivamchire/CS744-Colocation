import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(df, column1, column2, title, filename):
    # List of model names
    models = df['Model 1'].unique()

    # Create a DataFrame for the tuple strings and the minimum values
    df_tuples = pd.DataFrame(index=models, columns=models)
    df_min_values = pd.DataFrame(index=models, columns=models)

    # Fill the DataFrame with tuple strings and minimum values from the CSV file
    for _, row in df.iterrows():
        model1 = row["Model 1"]
        model2 = row["Model 2"]
        value1 = format(row[column1], '.3f')
        value2 = format(row[column2], '.3f')
        
        # Check if any of the data is None or NaN
        if pd.isnull(model1) or pd.isnull(model2) or pd.isnull(value1) or pd.isnull(value2):
            continue  # Skip to the next iteration

        df_tuples.loc[model1, model2] = f"({value1}, {value2})"
        # change this to whatever criterion is needed
        df_min_values.loc[model1, model2] = min(float(value1), float(value2))

    # Convert the minimum values to numbers
    df_min_values = df_min_values.astype(float)

    # Create a mask for the lower triangle excluding the diagonal
    # comment it out if the entire matrix is needed
    mask = np.tril(np.ones_like(df_min_values, dtype=bool), k=-1)

    # Plot the heatmap with the 'Paired' colormap and square cells
    plt.figure(figsize=(12, 10))
    # invert the colormap as needed in tput vs latency by adding _r at the end of cmap name
    # make sure to remove the mask if the entire matrix is needed
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

    plt.title(title, fontsize=20, y=1.08)
    plt.savefig(filename)

    plt.show()

# Read the CSV file
# Create a cleaner sheet with Model 1, Model 2, tput 1, tput 2, latency 1, latency 2 data for this
df = pd.read_csv("CS744-Readings - inf-inf heatmap.csv", sep=r'\s*,\s*', engine='python')

# Generate the throughput heatmap
generate_heatmap(df, "Normalized tput 1", "Normalized tput 2", "Normalized Throughput Matrix - Inference Colocation", "inf-inf heatmap_throughput.pdf")

# Generate the latency heatmap
generate_heatmap(df, "Normalized lat. 1", "Normalized lat. 2", "Normalized Latency Matrix - Inference Colocation", "inf-inf heatmap_latency.pdf")
