<<<<<<< HEAD
#from pdfminer.high_level import extract_text
=======
from pdfminer.high_level import extract_text
>>>>>>> 57bf30b63085dad699d6150f2a377862df51d7ce

import pandas as pd
import os, re

<<<<<<< HEAD
# def search_pdf_content(pdf_files_path, search_pattern):
#     """
#     Searches for a specified pattern within a list of PDF files and prints the matches.
#
#     :param pdf_files: List of PDF file paths
#     :param search_pattern: String or regex pattern to search for within the PDFs
#     """
#     pattern = re.compile(search_pattern)  # Compile the search pattern
#
#     for pdf_file in os.listdir(pdf_files_path):
#         # Check if the file exists
#         pdf_file = os.path.join(pdf_files_path, pdf_file)
#         if not os.path.exists(pdf_file):
#             print(f"File {pdf_file} does not exist.")
#             continue
#         try:
#             # Extract text from the PDF
#             text = extract_text(pdf_file)
#             matches = pattern.findall(text)  # Search for the pattern in the extracted text
#
#             if matches:
#                 print(f"Matches found in {pdf_file}:")
#                 for match in matches:
#                     print(match)
#             else:
#                 print(f"No matches found in {pdf_file}.")
#         except Exception as e:
#             print(f"An error occurred while processing {pdf_file}: {e}")
=======
def search_pdf_content(pdf_files_path, search_pattern):
    """
    Searches for a specified pattern within a list of PDF files and prints the matches.

    :param pdf_files: List of PDF file paths
    :param search_pattern: String or regex pattern to search for within the PDFs
    """
    pattern = re.compile(search_pattern)  # Compile the search pattern

    for pdf_file in os.listdir(pdf_files_path):
        # Check if the file exists
        pdf_file = os.path.join(pdf_files_path, pdf_file)
        if not os.path.exists(pdf_file):
            print(f"File {pdf_file} does not exist.")
            continue
        try:
            # Extract text from the PDF
            text = extract_text(pdf_file)
            matches = pattern.findall(text)  # Search for the pattern in the extracted text

            if matches:
                print(f"Matches found in {pdf_file}:")
                for match in matches:
                    print(match)
            else:
                print(f"No matches found in {pdf_file}.")
        except Exception as e:
            print(f"An error occurred while processing {pdf_file}: {e}")
>>>>>>> 57bf30b63085dad699d6150f2a377862df51d7ce


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# Similarity Function based on Levenshtein Distance
def levenshtein_similarity(s1, s2):
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1
    distance = levenshtein_distance(s1, s2)
    return 1 - distance / max_len


def preprocess_ptm_name(ptm):
    cleaned = ""
    splitted_content = str(ptm).split('/')
    if len(splitted_content) < 3:
        cleaned = splitted_content[1]
    elif len(splitted_content) == 3:
        cleaned = splitted_content[2]
    return cleaned


def get_most_freq_pipeline_tag(tuples_list):
    # Counting the frequency of elements in the second position of each tuple
    frequency = {}
    for _, element in tuples_list:
        if element in frequency:
            frequency[element] += 1
        else:
            frequency[element] = 1

    # Sorting the elements based on their frequency, from the most frequent to the least
    sorted_elements = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    return sorted_elements


def remove_formatting_characters(list_string):
    # Remove all formatting characters such as \n (newline), \t (tab), etc., by replacing them with a space
    # This can help in situations where you want to retain spaces but remove other types of whitespace
    cleaned_list = []
    for s in list_string:
        cleaned_string = re.sub(r'\s+', ' ', s)
        cleaned_list.append(cleaned_string)
    return cleaned_list

def get_most_freq_se_task(ptm_name, all_papers, taxonomy, macro_task):
    df_all = pd.read_csv(all_papers)
    df_task = pd.read_csv(taxonomy)
    df_macro = pd.read_csv(macro_task)
    ids = []
    for abstract, paper_id in zip(df_all['ABSTRACT'].values.astype(str), df_all['PAPER_ID']):
        if str(ptm_name).lower() in str(abstract).lower():
            ids.append(paper_id)
    se_sub_list = []
    for sub_task, list_id in zip(df_macro['SE task'],df_macro['Paper ID'].values.astype(str)):
        if list_id:
            aggr_list = str(list_id).split(',')
            for id in ids:
                if str(id) in aggr_list:
                    #print(sub_task)
                    se_sub_list.append(sub_task)
    macro_task_lists = []
    for macro, sub_task_list in zip(df_task['Macro-task'], df_task['Sub-tasks'].values.astype(str)):
        if sub_task_list:

            for map_sub in se_sub_list:
                if map_sub in sub_task_list:
                    macro_task_lists.append(macro)


    return macro_task_lists, se_sub_list


# Function to search for similar PTM names
def search_ptm_name(ptm, hf_dump):
    df_hf = pd.read_csv(hf_dump)
    similar_ptms = []
    for stored_ptm, tag in zip(df_hf['model_name'].values.astype(str), df_hf['tags']):
        stored_ptm = preprocess_ptm_name(stored_ptm)
        similarity_threshold = 0.7  # Adjust this threshold as needed

        similarity = levenshtein_similarity(ptm, stored_ptm)
        if similarity >= similarity_threshold:
            similar_ptms.append((stored_ptm, tag))

    return set(similar_ptms)
