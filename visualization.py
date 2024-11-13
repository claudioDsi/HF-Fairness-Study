import matplotlib.pyplot as plt
import pandas as pd
def aggregate_macro_task():
    # Mapping sub-tasks to their respective macro-tasks

    new_data = [
        ("Generating code patches", 1, [1]),
        ("Code comment generation", 4, [5, 7, 29, 33]),
        ("Bug fix/Program repair", 7, [7, 24, 35, 110, 111, 20, 30]),
        ("Injecting code mutants", 1, [7]),
        ("Assert statement generation", 1, [7]),
        ("Code summarization", 5, [10, 14, 23, 67, 20]),
        ("Code generation/completion", 14, [2, 4, 18, 24, 30, 48, 50, 51, 54, 57, 60, 77, 20, 30]),
        ("StackOverflow query reformulation", 1, [8]),
        ("Dynamic software update", 1, [9]),
        ("Algorithm classification", 1, [12]),
        ("Code clone detection", 4, [12, 41, 76, 126]),
        ("Code search", 4, [12, 30, 39, 65]),
        ("Program syntesis", 1, [13]),
        ("Automated commit generation", 4, [16, 70, 49, 119]),
        ("Bug report", 1, [17, 112]),
        ("Test oracle generation", 1, [19]),
        ("Code review", 5, [21, 58, 59, 105, 121]),
        ("Requirement classification", 2, [25, 78]),
        ("Smart contract generation", 2, [26, 73]),
        ("Vulnerability fix", 2, [27, 40]),
        ("API reviews classification", 1, [31]),
        ("StackOverflow title generation", 3, [32, 47, 72]),
        ("Vulnerability detection", 1, [36, 126]),
        ("Story point estimation", 1, [37]),
        ("Code translation", 6, [41, 62, 106, 109, 20, 30]),
        ("Code-to-code search", 1, [41]),
        ("Sentiment analysis", 5, [3, 42, 124, 125, 127]),
        ("Issue report classification", 1, [68, 117]),
        ("String generation secondary studies", 1, [71]),
        ("Pull request description generation", 1, [43]),
        ("Commit classification", 2, [44, 49]),
        ("Automatic fault localization", 1, [45]),
        ("StackOverflow post classification", 1, [45]),
        ("Program merge", 1, [53]),
        ("Performance improvements", 1, [55]),
        ("Issue title generation", 1, [56]),
        ("Signal temporal logic requirements", 1, [22]),
        ("Function naming", 1, [24]),
        ("Binary Code Matching", 1, [34]),
        ("Text summarization", 1, [74]),
        ("Language translation", 1, [75]),
        ("Low-code generation", 1, [107]),
        ("Graphical model completion", 1, [108]),
        ("GUI testing", 1, [113]),
        ("Stack overflow post summarization", 1, [114]),
        ("Testing generation", 1, [115]),
        ("App-reviews classification and feedback generation", 1, [118]),
        ("Technical debt detection", 1, [123]),
        ("Classifying UML diagrams", 1, [128]),
        ("Software library recommendation", 1, [129]),
        ("Traceability", 1, [132])
    ]


    macro_tasks = {
        "Code-related task": [
            "Code summarization", "Code generation/completion", "Dynamic software update",
            "Algorithm classification", "Code clone detection", "Code search", "Code translation",
            "Code-to-code search", "Program syntesis", "Injecting code mutants", "Program merge",
            "Function naming", "Performance improvements", "Binary Code Matching", "Technical debt detection",
            "Software library recommendation"
        ],
        "Testing/Program repair": [
            "Generating code patches", "Bug fix/Program repair", "Vulnerability fix",
            "Test oracle generation", "Vulnerability detection", "Automatic fault localization",
            "GUI testing", "Testing generation", "Assert statement generation"
        ],
        "Documentation/Requirements": [
            "Code comment generation", "Automated commit generation", "API reviews classification",
            "Requirement classification", "Code review", "Bug report", "Signal temporal logic requirements",
            "Text summarization", "Story point estimation", "App-reviews classification and feedback generation",
            "Traceability"
        ],
        "StackOverflow related": [
            "StackOverflow query reformulation", "StackOverflow title generation",
            "StackOverflow post classification", "Stack overflow post summarization"
        ],
        "Github-related": [
            "Issue report classification", "Pull request description generation", "Commit classification",
            "Issue title generation"
        ],
        "Miscellaneous": [
            "Smart contract generation", "Sentiment analysis", "String generation secondary studies",
            "Language translation", "Low-code generation", "Graphical model completion", "Classifying UML diagrams"
        ]
    }

    # Initialize a dictionary to hold the support count for each macro-task
    macro_task_support = {key: 0 for key in macro_tasks.keys()}

    # Sum the support for each macro-task based on the sub-tasks
    for task, support, _ in new_data:
        for macro_task, sub_tasks in macro_tasks.items():
            if task in sub_tasks:
                macro_task_support[macro_task] += support




def plot_dataset_stats(input_file):
    # data = {
    #     'reinforcement-learning': 29970,
    #     'text-classification': 19418,
    #     'text-generation': 16615,
    #     'text-to-image': 11142,
    #     'text2text-generation': 10193,
    #     'automatic-speech-recognition': 8668,
    #     'token-classification': 7831,
    #     'fill-mask': 5267,
    #     'image-classification': 5105,
    #     'question-answering': 4462,
    #     'audio-to-audio': 3583,
    #     'translation': 2952,
    #     'conversational': 2586,
    #     'sentence-similarity': 2520,
    #     'feature-extraction': 2212,
    #     'text-to-speech': 1651,
    #     'summarization': 1403,
    #     'audio-classification': 951,
    #     'unconditional-image-generation': 865,
    #     'object-detection': 521,
    #     'image-segmentation': 300,
    #     'image-to-text': 299,
    #     'video-classification': 245,
    #     'multiple-choice': 235,
    #     'image-to-image': 204,
    #     'zero-shot-classification': 192,
    #     'tabular-classification': 183,
    #     'zero-shot-image-classification': 130,
    #     'tabular-regression': 89,
    #     'table-question-answering': 62,
    #     'text-to-audio': 52,
    #     'depth-estimation': 50,
    #     'visual-question-answering': 50,
    #     'text-to-video': 48,
    #     'document-question-answering': 46,
    #     'voice-activity-detection': 18,
    #     'graph-ml': 11,
    #     'other': 10,
    #     'robotics': 9,
    #     'time-series-forecasting': 1
    # }
    df = pd.read_csv(input_file)
    categories = df['tags'].tolist()  # Extract categories from the DataFrame
    values = df['count'].tolist()  # Extract values for each category
    labels = categories  # Use categories as labels
    # categories = list(data.keys())  # Extract categories from the dictionary
    # values = [data[cat] for cat in categories]  # Extract values for each category
    # labels = categories  # Use categories as labels

    # Plotting setup with increased figure size for better readability
    plt.figure(figsize=(24, 20))  # Further increase the figure size

    plt.barh(labels, values, color='darkblue')  # Using horizontal bar plot
    plt.xlabel('Support', fontsize=40)  # Increase font size for x-axis label
    plt.title('Pipeline tag', fontsize=40)  # Increase font size for title

    # Rotate x-ticks for better visibility and increase font size
    plt.xticks(rotation=45, fontsize=30)  # Increase font size for x-ticks
    plt.yticks(fontsize=30)  # Increase font size for y-ticks

    # Find the maximum value in the list and add some extra space for readability
    max_value = max(values)
    extra_space = max_value * 0.2  # Add 10% extra space
    plt.xlim(0, max_value + extra_space)  # Adjust x-axis limits

    # Adding value labels to each bar
    for index, value in enumerate(values):
        plt.text(value, index, f'{value:.0f}', va='center', fontsize=30)  # Adjust font size for bar labels

    plt.tight_layout()  # Adjust layout to make room for the rotated x-tick labels

    # Save the plot as a PDF file
    plt.savefig('stats/d2_distribution.pdf', format='pdf')

    #plt.show()  # Display the plot


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


plot_bar_chart()

#plot_bar_chart('stats/d2_stats.csv')