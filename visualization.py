import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv



def plot_dataset_stats(input_file, x, y, out):

    df = pd.read_csv(input_file)
    categories = df[x].tolist()  # Extract categories from the DataFrame
    values = df[y].tolist()  # Extract values for each category
    labels = []

    for cat in categories:
        if '/' in cat:
            labels.append(cat.split('/')[0])

    #labels =categories

    # Use categories as labels
    # categories = list(data.keys())  # Extract categories from the dictionary
    # values = [data[cat] for cat in categories]  # Extract values for each category
    # labels = categories  # Use categories as labels

    # Plotting setup with increased figure size for better readability
    plt.figure(figsize=(20, 12))  # Further increase the figure size

    plt.bar(labels, values, color='darkblue')  # Using horizontal bar plot
   # plt.xlabel('Tags', fontsize=20)  # Increase font size for x-axis label
    #plt.title('Stars', fontsize=20)  # Increase font size for title

    # Rotate x-ticks for better visibility and increase font size
    plt.xticks(rotation=45, fontsize=30)  # Increase font size for x-ticks
    plt.yticks(fontsize=40)  # Increase font size for y-ticks


    plt.tight_layout()  # Adjust layout to make room for the rotated x-tick labels

    # Save the plot as a PDF file
    plt.savefig(out, format='pdf')

    plt.show()  # Display the plot


def plot_dataset_stats_2(input_file, x, y1, y2, out):
    """
    Plots dataset statistics, displaying two metrics (e.g., Stars and Forks) for each category.

    Parameters:
        input_file (str): Path to the CSV file.
        x (str): Column name for categories.
        y1 (str): Column name for the first metric (e.g., Stars).
        y2 (str): Column name for the second metric (e.g., Forks).
        out (str): Output file path for the plot (PDF format).
    """
    # Load the dataset
    df = pd.read_csv(input_file)

    # Extract data from the DataFrame
    categories = df[x].tolist()
    values1 = df[y1].tolist()  # First metric (e.g., Stars)
    values2 = df[y2].tolist()  # Second metric (e.g., Forks)
    labels = []

    # Generate simplified labels if '/' exists in category names
    for cat in categories:
        if '/' in cat:
            labels.append(cat.split('/')[0])
        else:
            labels.append(cat)

    # Plotting setup
    plt.figure(figsize=(20, 12))  # Increase figure size for better readability

    # Define bar positions
    x_positions = np.arange(len(labels))  # Position of categories on the x-axis
    bar_width = 0.4  # Width of each bar

    # Create bars for both metrics
    plt.bar(x_positions - bar_width / 2, values1, bar_width, color='darkblue', label='Stars')
    plt.bar(x_positions + bar_width / 2, values2, bar_width, color='red', label='Forks')

    # Customizations
    plt.xticks(x_positions, labels, rotation=45, fontsize=30)  # Rotate x-tick labels
    plt.yticks(fontsize=40)  # Adjust font size for y-ticks
    plt.legend(fontsize=40)  # Add legend and set its font size
    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save the plot as a PDF file
    plt.savefig(out, format='pdf')

    plt.show()  # Display the plot





def plot_bar_chart(label_fontsize=18, tick_fontsize=24):
    data = {
        'M1': 'Code-related tasks',
        'M2': 'Program repair',
        'M3': 'Documentation support',
        'M4': 'Classification of SE artifacts',
        'M5': 'Text-engineering tasks',
        'M6': 'Miscellaneous'
    }
    # Values for each category
    values = [33, 16, 22, 8, 8, 12]

    # Abbreviations for x-axis labels
    abbreviations = list(data.keys())

    # Generating a list of gray colors
    num_bars = len(data)
    gray_colors = [str(i / num_bars) for i in range(num_bars)]

    # Plotting
    plt.figure(figsize=(9, 6.5))
    bars = plt.bar(abbreviations, values, color=gray_colors)

    plt.xticks(rotation=45, fontsize=tick_fontsize)  # Rotate x-axis labels
    plt.yticks(fontsize=tick_fontsize)

    for i, value in enumerate(values):
        plt.text(i, value + 0.1, str(value), ha='center', va='bottom', fontsize=label_fontsize)

    # Creating legend
    legend_labels = [f'{abbr} - {data[abbr]}' for abbr in abbreviations]
    plt.legend(bars, legend_labels, loc='upper right', fontsize='x-large')

    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.savefig('macro_stats.pdf')  # Export the figure to a PDF file
    plt.show()


# Remember to call the function to generate the plot
def csv_to_latex(input_csv, columns_to_drop, output_tex):
    """
    Reads a CSV file, drops specified columns, and exports the result to a LaTeX table.

    Parameters:
    input_csv (str): Path to the input CSV file.
    columns_to_drop (list): List of column names to drop.
    output_tex (str): Path to save the LaTeX table.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_csv)

        # Drop specified columns
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Export DataFrame to LaTeX format
        latex_table = df.to_latex(index=False)

        # Save the LaTeX table to a file
        with open(output_tex, 'w') as f:
            f.write(latex_table)

        print(f"LaTeX table successfully saved to {output_tex}.")
    except Exception as e:
        print(f"An error occurred: {e}")

#plot_dataset_stats_2("stats/gh_code_ranked_by_stars.csv","Repository", "Stars", "Forks","stats/stars.pdf")

#plot_bar_chart('stats/d2_stats.csv')

#dump_categories()

plot_long_tail_distribution_grouped("MSR-paper/filtered.csv")
#csv_to_latex("matched_model.csv", ["Matched Keywords First Set","Matched Keywords Second Set"], "output_table.tex")