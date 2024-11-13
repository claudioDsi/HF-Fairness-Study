import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config as cf
import  time
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt






def run_classifier(dataset,desc, cat, model, results_csv_path):

    data = pd.read_csv(dataset)

    # Assuming the columns are named 'card_data' and 'tags'
    X = data[desc]
    y = data[cat]

    # Vectorize the text data using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # K-Fold validation parameters
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize LinearSVC model
    if model == 'SVC':
        model = LinearSVC(random_state=42)
    elif model == 'CNB':
        model = ComplementNB()

    results = []

    # Perform K-Fold validation
    results = []
    classification_reports = []
    fold = 1
    for train_index, test_index in kf.split(X_vectorized):
        X_train, X_test = X_vectorized[train_index], X_vectorized[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Start time for training
        start_time = time.time()

        # Train model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)

        # End time for prediction
        end_time = time.time()

        # Compute total time taken
        time_taken = end_time - start_time

        #report = classification_report(y_test, y_pred, output_dict=True)

        #print(classification_report(y_test, y_pred))
        #df_report = pd.DataFrame(report).transpose()

        # Add fold number as a column for identification
        #df_report['Fold'] = fold

        #classification_reports.append(df_report)

        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)


<<<<<<< HEAD
=======



>>>>>>> 57bf30b63085dad699d6150f2a377862df51d7ce
        # Append results for this fold, including time computation
        results.append({
            'Fold': fold,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Time (seconds)': time_taken
        })

        fold += 1

    # Convert results list to DataFrame


    df_results = pd.DataFrame(results)


    # Calculate averages and create a DataFrame for them
    average_values = {
        'Fold': 'Average',
        'Precision': df_results['Precision'].mean(),
        'Recall': df_results['Recall'].mean(),
        'F1 Score': df_results['F1 Score'].mean(),
        'Time (seconds)': df_results['Time (seconds)'].mean()
    }
    df_average = pd.DataFrame([average_values])

    # Concatenate the average row to the original DataFrame
    df_final_results = pd.concat([df_results, df_average], ignore_index=True)

    # Save the combined DataFrame to a CSV file
    df_final_results.to_csv(results_csv_path, index=False)
    #df_final_results.to_csv(results_csv_path, index=False)

    print(f'Results have been saved to {results_csv_path}')

    #df_all_reports = pd.concat(classification_reports, ignore_index=True)

    # Pivot table to have a better layout for LaTeX conversion
    #df_pivot_reports = df_all_reports.pivot_table(index=['Fold'],
                                                  # margins=True,
                                                  # margins_name='Average',
                                                  # aggfunc='mean')

    # Convert the final DataFrame to LaTeX
    #latex_code = df_pivot_reports.to_latex()

    # # Export the LaTeX code to a .tex file
    # with open('classification_report_table.tex', 'w') as f:
    #     f.write(latex_code)
    #
    # # Optionally, you can print the LaTeX code or the DataFrame for review
    # print(latex_code)
    #
