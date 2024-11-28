from operator import index
from collections import Counter
import pandas as pd
import re
import config as cf
from github import Github, Auth, GithubException
from collections import defaultdict
import requests
from textdistance import levenshtein
import yaml
import os
import json
from time import sleep
import csv
import re







def save_python_file_names_to_csv(json_file_path, output_csv_path):
    """
    Save the names of Python files identified in the JSON file to a CSV file.

    Parameters:
        json_file_path (str): The path to the JSON file.
        output_csv_path (str): The path to the output CSV file.

    Returns:
        None
    """
    try:
        # Load the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        python_files = []

        # Loop through the JSON structure
        for key, entries in data.items():  # Iterates over the top-level keys
            for entry in entries:  # Iterates over each list of entries
                if entry.get("model_mentions_in_code"):  # Check if model_mentions_in_code is True
                    for file_info in entry.get("files", []):  # Check the files array
                        if file_info["file_name"].endswith('.py'):  # Check for Python file
                            python_files.append(file_info["file_name"])

        # Write the Python file names to a CSV file
        with open(output_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write header
            csvwriter.writerow(["Python File Name"])
            # Write file names
            for file_name in python_files:
                csvwriter.writerow([file_name])

        print(f"Python file names have been saved to {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {json_file_path} is not a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def find_most_common_python_file_names(json_file_path, top_n=10):
    """
    Find the most common Python file names in the JSON file.

    Parameters:
        json_file_path (str): The path to the JSON file.
        top_n (int): The number of most common file names to return.

    Returns:
        list: A list of tuples with the most common file names and their counts.
    """
    try:
        # Load the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        python_files = []

        # Loop through the JSON structure
        for key, entries in data.items():  # Iterates over the top-level keys
            for entry in entries:  # Iterates over each list of entries
                if entry.get("model_mentions_in_code"):  # Check if model_mentions_in_code is True
                    for file_info in entry.get("files", []):  # Check the files array
                        if file_info["file_name"].endswith('.py'):  # Check for Python file
                            python_files.append(file_info["file_name"])

        # Count occurrences of each file name
        file_name_counts = Counter(python_files)

        # Return the most common file names
        return file_name_counts.most_common(top_n)

    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file {json_file_path} is not a valid JSON.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []




def count_python_files_by_repository_to_csv(json_file_path, output_csv_path):
    """
    Count the total number of Python files where 'model_mentions_in_code' is true,
    grouped by each repository, and save the result to a CSV file.

    Parameters:
        json_file_path (str): The path to the JSON file.
        output_csv_path (str): The path to the output CSV file.

    Returns:
        None
    """
    try:
        # Load the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Dictionary to store counts by repository
        repository_file_counts = {}

        # Loop through the JSON structure
        for key, entries in data.items():  # Iterates over the top-level keys
            for entry in entries:  # Iterates over each list of entries
                if entry.get("model_mentions_in_code"):  # Check if model_mentions_in_code is True
                    repo_name = entry.get("repository", "Unknown Repository")
                    python_file_count = sum(
                        1 for file_info in entry.get("files", [])
                        if file_info["file_name"].endswith('.py')
                    )
                    # Update the count for the repository
                    repository_file_counts[repo_name] = (
                            repository_file_counts.get(repo_name, 0) + python_file_count
                    )

        # Write the results to a CSV file
        with open(output_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write header
            csvwriter.writerow(["Repository", "Python File Count"])
            # Write repository data
            for repo, count in repository_file_counts.items():
                csvwriter.writerow([repo, count])

        print(f"Results have been saved to {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {json_file_path} is not a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def split_csv(input_file, output_dir, rows_per_file):
    """
    Splits a large CSV file into smaller files with a specified number of rows each.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_dir (str): Directory where the split files will be saved.
    - rows_per_file (int): Number of rows per output file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file in chunks to handle large files
    chunk_iter = pd.read_csv(input_file, chunksize=rows_per_file)

    for i, chunk in enumerate(chunk_iter):
        # Construct the output file name
        output_file = os.path.join(output_dir, f"split_part_{i + 1}.csv")

        # Save the current chunk to a new file
        chunk.to_csv(output_file, index=False)

        print(f"Saved {output_file} with {len(chunk)} rows.")


def build_terms_from_file(file_path):
    """
    Reads terms from a text file and constructs a regex pattern.

    Args:
        file_path (str): Path to the text file containing terms.

    Returns:
        str: A regex pattern that matches any of the terms in the file.
    """
    try:
        with open(file_path, 'r') as file:
            # Read lines and clean up whitespace
            terms = [line.strip() for line in file if line.strip()]

        # Join the terms with '|' to create a regex OR pattern
        regex_pattern = r'(' + r' | '.join(re.escape(term) for term in terms) + r')'
        return regex_pattern
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ''
    except Exception as e:
        print(f"An error occurred: {e}")
        return ''


def search_for_libraries(content):
    """
    Searches for terms in the provided content and counts occurrences for specific term sets.
    """
    count_first = 0


    set_keywords_first = r'(aif360 | fairlearn | fklearn)'


    first_set = re.search(set_keywords_first, content, re.IGNORECASE)


    if first_set:
        count_first += 1


    return count_first



def search_for_terms(content):
    """
    Searches for terms in the provided content and counts occurrences for specific term sets.
    """
    count_first = 0
    count_second = 0
    countTot = 0

    #set_keywords_first = r'(fairness | bias | ethics | morality | responsibility | fair)'
    set_keywords_first = build_terms_from_file("Gao_work/extra_keywords_paragraph.txt")
    set_keywords_second = r'(toolkit | audit | testing | assessment | accountability)'

    first_set = re.search(set_keywords_first, content, re.IGNORECASE)
    second_set = re.search(set_keywords_second, content, re.IGNORECASE)

    if first_set:
        count_first += 1

    if second_set:
        count_second += 1

    if first_set and second_set:
        countTot += 1



    return count_first, count_second, countTot


def search_keywords_and_libraries(json_file, output_csv):
    """
    Searches for specific terms in the README content and specific Python libraries
    in the source code files stored in the JSON file. Also counts model mentions
    in the code and computes the average stars and forks for repositories.

    :param json_file: Path to the JSON file containing mined data.
    :param libraries: A list of Python libraries to search for in the source code.
    :param output_csv: Path to the output CSV file for storing summary results.
    :return: A dictionary summarizing the search results.
    """
    # Load the mined data from the JSON file
    with open(json_file, 'r', encoding='utf-8') as file:
        mined_data = json.load(file)

    # Initialize variables for CSV reporting
    csv_data = []
    total_repositories = 0
    total_model_mentions = 0
    total_stars = 0
    total_forks = 0
    repo_count_with_stars = 0

    # Initialize the results dictionary
    search_results = {}

    # Iterate over each model in the mined data
    for model_name, repositories in mined_data.items():
        print(f"Analyzing model: {model_name}")
        model_summary = []
        model_mentions_count = 0

        # Iterate over each repository for the current model
        for repo_data in repositories:
            repo_name = repo_data["repository"]
            readme_content = repo_data.get("readme", "").lower()
            files = repo_data.get("files", [])
            stars = repo_data.get("stars", 0)
            forks = repo_data.get("forks", 0)
            total_stars += stars
            total_forks += forks
            if stars > 0 or forks > 0:
                repo_count_with_stars += 1

            # Check for model mentions in code
            model_mentions = repo_data.get("model_mentions_in_code", False)
            if model_mentions:
                model_mentions_count += 1

            # Search for terms in the README content
            count_first, count_second, countTot = search_for_terms(readme_content)

            # Search for libraries in the source code files
            # matched_libraries = []
            # for file_data in files:
            #     file_content = file_data["content"].lower()
            #     for library in libraries:
            #         if library.lower() in file_content and library not in matched_libraries:
            #             matched_libraries.append(library)

            matched_libraries = []
            count_lib = 0
            for file_data in files:
                file_content = file_data["content"].lower()
                count_lib+=search_for_libraries(file_content)

            # Summarize findings for this repository
            repo_summary = {
                "repository": repo_name,
                "matched_keywords_first_set": count_first,
                "matched_keywords_second_set": count_second,
                "matched_keywords_total": countTot,
                "matched_libraries": count_lib
            }
            model_summary.append(repo_summary)

            # Add to CSV data
            csv_data.append({
                "Model": model_name,
                "Repository": repo_name,
                "Matched Keywords First Set": count_first,
                "Matched Keywords Second Set": count_second,
                "Matched Keywords Total": countTot,
                "Matched Libraries": len(matched_libraries),
                "Stars": stars,
                "Forks": forks,
                "Model Mentions in Code": int(model_mentions)
            })

        # Add the summary for this model to the results
        search_results[model_name] = model_summary
        total_model_mentions += model_mentions_count
        total_repositories += len(repositories)

    # Compute averages
    average_stars = total_stars / repo_count_with_stars if repo_count_with_stars > 0 else 0
    average_forks = total_forks / repo_count_with_stars if repo_count_with_stars > 0 else 0

    # Print the search results
    print("Search completed. Summary of findings:")
    print(f"Total repositories analyzed: {total_repositories}")
    print(f"Total model mentions in code: {total_model_mentions}")
    print(f"Average stars: {average_stars:.2f}")
    print(f"Average forks: {average_forks:.2f}")

    # Write results to a CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Model", "Repository", "Matched Keywords First Set",
                      "Matched Keywords Second Set", "Matched Keywords Total",
                      "Matched Libraries", "Stars", "Forks", "Model Mentions in Code"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_data)

    return search_results



def filter_models_with_code(input_csv, output_csv):
    """
    Filters models that have at least one matched keyword or matched library from the input CSV
    and writes them to a new output CSV.

    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to the output CSV file.
    :return: None
    """
    # Read the input CSV and filter rows
    filtered_rows = []
    with open(input_csv, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        # Check for matching conditions
        for row in reader:
            if int(row["Model Mentions in Code"]) > 0:
                filtered_rows.append(row)

    # Write the filtered rows to the output CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Filtered models have been saved to {output_csv}")

def filter_models_with_matches_updated(input_csv, output_csv):
    """
    Filters models that have at least one matched keyword (either set) or matched library
    from the input CSV and writes them to a new output CSV.

    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to the output CSV file.
    :return: None
    """
    # Read the input CSV and filter rows


    #     filtered_rows.append(row)
    filtered_rows = []
    with open(input_csv, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        # Check for matching conditions
        for row in reader:
            if (  int(row["Matched Keywords Total"]) > 0 or
                    int(row["Matched Libraries"]) > 0):
                filtered_rows.append(row)

    # Write the filtered rows to the output CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Filtered models have been saved to {output_csv}")






def merge_json_files(json_file1, json_file2, output_file):
    """
    Merges two JSON files containing repository information in the given format.

    :param json_file1: Path to the first JSON file.
    :param json_file2: Path to the second JSON file.
    :param output_file: Path to the output JSON file for storing merged results.
    :return: None
    """
    # Load both JSON files
    with open(json_file1, 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    with open(json_file2, 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)

    # Initialize a dictionary to hold merged data
    merged_data = defaultdict(list)

    # Merge the data from the first file
    for model_name, repos in data1.items():
        merged_data[model_name].extend(repos)

    # Merge the data from the second file, ensuring no duplicates
    for model_name, repos in data2.items():
        for repo in repos:
            repo_names = [r["repository"] for r in merged_data[model_name]]
            if repo["repository"] not in repo_names:
                merged_data[model_name].append(repo)
            else:
                # Merge details of the repository if it already exists
                existing_repo = next(r for r in merged_data[model_name] if r["repository"] == repo["repository"])
                existing_repo["stars"] = max(existing_repo.get("stars", 0), repo.get("stars", 0))
                existing_repo["forks"] = max(existing_repo.get("forks", 0), repo.get("forks", 0))
                existing_repo["readme"] = existing_repo.get("readme", "") or repo.get("readme", "")
                existing_repo["model_mentions_in_code"] = existing_repo.get("model_mentions_in_code",
                                                                            False) or repo.get("model_mentions_in_code",
                                                                                               False)

                # Merge files
                existing_file_names = [file["file_name"] for file in existing_repo.get("files", [])]
                for file in repo.get("files", []):
                    if file["file_name"] not in existing_file_names:
                        existing_repo["files"].append(file)

    # Save the merged data to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, indent=4)

    print(f"Merged JSON data has been saved to {output_file}")


def search_keywords_and_libraries_old(json_file, keywords, libraries, output_csv):
    """
    This function searches for specific keywords in the README content
    and specific Python libraries in the source code files stored in the JSON file.
    It also counts the number of model mentions in the code and computes
    the average stars and forks for repositories.

    :param json_file: Path to the JSON file containing mined data.
    :param keywords: A list of keywords to search for in the README content.
    :param libraries: A list of Python libraries to search for in the source code.
    :param output_csv: Path to the output CSV file for storing summary results.
    :return: A dictionary summarizing the search results.
    """
    # Load the mined data from the JSON file
    with open(json_file, 'r', encoding='utf-8') as file:
        mined_data = json.load(file)

    # Initialize variables for CSV reporting
    csv_data = []
    total_repositories = 0
    total_model_mentions = 0
    total_stars = 0
    total_forks = 0
    repo_count_with_stars = 0

    # Initialize the results dictionary
    search_results = {}

    # Iterate over each model in the mined data
    for model_name, repositories in mined_data.items():
        print(f"Analyzing model: {model_name}")
        model_summary = []
        model_mentions_count = 0

        # Iterate over each repository for the current model
        for repo_data in repositories:
            repo_name = repo_data["repository"]
            readme_content = repo_data.get("readme", "").lower()
            files = repo_data.get("files", [])
            stars = repo_data.get("stars", 0)
            forks = repo_data.get("forks", 0)
            total_stars += stars
            total_forks += forks
            if stars > 0 or forks > 0:
                repo_count_with_stars += 1

            # Check for model mentions in code
            model_mentions = repo_data.get("model_mentions_in_code", False)
            if model_mentions:
                model_mentions_count += 1

            # Search for keywords in the README content
            matched_keywords = [keyword for keyword in keywords if keyword.lower() in readme_content]
            #search_for_terms(readme_content)

            # Search for libraries in the source code files
            matched_libraries = []
            for file_data in files:
                file_content = file_data["content"].lower()
                for library in libraries:
                    if library.lower() in file_content and library not in matched_libraries:
                        matched_libraries.append(library)

            # Summarize findings for this repository
            repo_summary = {
                "repository": repo_name,
                "matched_keywords": matched_keywords,
                "matched_libraries": matched_libraries
            }
            model_summary.append(repo_summary)

            # Add to CSV data
            csv_data.append({
                "Model": model_name,
                "Repository": repo_name,
                "Matched Keywords": len(matched_keywords),
                "Matched Libraries": len(matched_libraries),
                "Stars": stars,
                "Forks": forks,
                "Model Mentions in Code": int(model_mentions)
            })

        # Add the summary for this model to the results
        search_results[model_name] = model_summary
        total_model_mentions += model_mentions_count
        total_repositories += len(repositories)

    # Compute averages
    average_stars = total_stars / repo_count_with_stars if repo_count_with_stars > 0 else 0
    average_forks = total_forks / repo_count_with_stars if repo_count_with_stars > 0 else 0

    # Print the search results
    print("Search completed. Summary of findings:")
    print(f"Total repositories analyzed: {total_repositories}")
    print(f"Total model mentions in code: {total_model_mentions}")
    print(f"Average stars: {average_stars:.2f}")
    print(f"Average forks: {average_forks:.2f}")

    # Write results to a CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Model", "Repository", "Matched Keywords", "Matched Libraries", "Stars", "Forks", "Model Mentions in Code"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_data)

    return search_results


def search_keywords_and_libraries_old(json_file, keywords, libraries):
    """
    This function searches for specific keywords in the README content
    and specific Python libraries in the source code files stored in the JSON file.

    :param json_file: Path to the JSON file containing mined data.
    :param keywords: A list of keywords to search for in the README content.
    :param libraries: A list of Python libraries to search for in the source code.
    :return: A dictionary summarizing the search results.
    """
    # Load the mined data from the JSON file
    with open(json_file, 'r', encoding='utf-8') as file:
        mined_data = json.load(file)

    # Initialize the results dictionary
    search_results = {}

    # Iterate over each model in the mined data
    for model_name, repositories in mined_data.items():
        print(f"Analyzing model: {model_name}")
        model_summary = []

        # Iterate over each repository for the current model
        for repo_data in repositories:
            repo_name = repo_data["repository"]
            readme_content = repo_data.get("readme", "").lower()
            files = repo_data.get("files", [])

            # Search for keywords in the README content
            matched_keywords = [keyword for keyword in keywords if keyword.lower() in readme_content]

            # Search for libraries in the source code files
            matched_libraries = []
            for file_data in files:
                file_content = file_data["content"].lower()
                for library in libraries:
                    if library.lower() in file_content and library not in matched_libraries:
                        matched_libraries.append(library)

            # Summarize findings for this repository
            repo_summary = {
                "repository": repo_name,
                "matched_keywords": matched_keywords,
                "matched_libraries": matched_libraries
            }
            model_summary.append(repo_summary)

        # Add the summary for this model to the results
        search_results[model_name] = model_summary

    # Print the search results
    print("Search completed. Summary of findings:")
    for model_name, summaries in search_results.items():
        print(f"\nModel: {model_name}")
        for summary in summaries:
            print(f"  Repository: {summary['repository']}")
            print(f"    Matched Keywords: {summary['matched_keywords']}")
            print(f"    Matched Libraries: {summary['matched_libraries']}")

    return search_results


def preprocessing_queries(models: list) -> list:
    with open('all_models_source_code_part_1.json', 'r', encoding="utf8", errors="ignore") as file:
        data = json.load(file)

    # Estraggo i modelli che ho già processato
    #models_with_files = [key for key, value in data.items() if len(dict(value)) != 0]

    # Li tolgo da quelli che devo ancora minare
    models = [item for item in models if item not in data.keys()]

    return models


def mine_github_source_code(models_name, token, output_file):
    """
    This function mines GitHub repositories for specific model names, searching
    for occurrences of the model name in the README and source code files.
    Additionally, it counts mentions of specific Python libraries (`aif360`, `fairlearn`, `fairkit-learn`)
    and saves metadata such as the number of stars, forks, and the URL of the repository.
    """
    # Initialize GitHub API client with the provided token
    g = Github(token)

    # Define the language filter
    language_filter = " language:Python"

    # Libraries to search for
    target_libraries = ["aif360", "fairlearn", "fklearn"]

    # Initialize the results dictionary
    mined_data = {}

    # Iterate over the model names
    for model_name in models_name:
        try:
            query = model_name + language_filter
            print(f"Searching for repositories related to model: {model_name}")

            # Search for repositories using the query
            repositories = g.search_repositories(query=query)

            # Initialize a list to store data for this model
            model_data = []

            # Iterate over the top 20 repositories
            for repo in repositories:
                repo_name = repo.full_name
                print(f"Mining repository: {repo_name}")

                # Initialize repository data
                repo_data = {
                    "repository": repo_name,
                    "url": repo.html_url,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "files": [],  # Store file URLs instead of content
                    "readme": "",
                    "model_mentions_in_code": False,
                    "library_mentions": {lib: 0 for lib in target_libraries}  # Initialize counts for target libraries
                }

                try:
                    # Fetch the README content if available
                    try:
                        readme_content = repo.get_readme().decoded_content.decode('utf-8')
                        repo_data["readme"] = readme_content
                        print(f"Fetched README for: {repo_name}")

                        # Check if the model name is mentioned in the README
                        if model_name.lower() in readme_content.lower():
                            print(f"Model name '{model_name}' found in README.")
                    except Exception:
                        print(f"No README found for: {repo_name}")

                    # Get the list of contents at the root of the repository
                    contents = repo.get_contents("")

                    # Fetch Python files and search for the model name and target libraries in the content
                    while contents:
                        file_content = contents.pop(0)
                        if file_content.type == 'file' and file_content.path.endswith(".py"):
                            print(f"Fetching file: {file_content.path}")

                            # Add the file URL to the files list
                            repo_data["files"].append({
                                "file_name": file_content.path,
                                "url": file_content.html_url
                            })

                            try:
                                # Decode file content
                                file_data = file_content.decoded_content.decode('utf-8')

                                # Check if the model name is mentioned in the source code
                                if model_name.lower() in file_data.lower():
                                    repo_data["model_mentions_in_code"] = True
                                    print(f"Model name '{model_name}' found in source file: {file_content.path}")

                                # Count mentions of target libraries
                                for lib in target_libraries:
                                    count = file_data.lower().count(lib.lower())
                                    if count > 0:
                                        repo_data["library_mentions"][lib] += count
                                        print(f"Library '{lib}' mentioned {count} time(s) in file: {file_content.path}")

                            except Exception:
                                print(f"Error reading file: {file_content.path}")

                except Exception as e:
                    print(f"Error while accessing repository '{repo_name}': {e}")

                # Add the repository data to the model's data
                model_data.append(repo_data)

                # Save incremental data for this repository
                mined_data[model_name] = model_data
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    json.dump(mined_data, json_file, ensure_ascii=False, indent=4)

                # To avoid hitting the GitHub API rate limit
                sleep(10)

        except Exception as e:
            print(f"Error while processing model name '{model_name}': {e}")

    print(f"Source code mining completed. Data saved to '{output_file}'.")


def search_fairness_keywords(df, columns, output_path):
    # Define the keywords to search for
    keywords = ["ethics", "fairness", "accountability", "transparency", "morality", "responsibility"]
    new_keywords = ["bias", "fairness", "assessment", "transparency", "accountability"]

    # Combine both lists of keywords
    all_keywords = set(keywords + new_keywords)

    # Create a filter mask for the rows containing any of the keywords
    mask = df[columns].apply(lambda row: any(keyword.lower() in str(row).lower() for keyword in all_keywords), axis=1)

    # Filter the rows matching the keywords
    filtered_rows = df[mask]

    # Export the filtered results to a CSV file
    filtered_rows.to_csv(output_path, index=False)

    print(f"Results have been saved to {output_path}")
    return filtered_rows


def split_df_by_tags(df, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Explode the tags column to create a row for each tag
    df_exploded = df.explode('tags')

    # Get the unique list of tags
    unique_tags = df_exploded['tags'].unique()

    # Iterate through each unique tag and create a separate dataset
    for tag in unique_tags:
        # Filter the DataFrame for rows containing the current tag
        tag_df = df_exploded[df_exploded['tags'] == tag]
        tag_df= tag_df.sort_values(by='downloads', ascending=False)


        # Define the output filename based on the tag (clean up the tag name for filename compatibility)
        file_name = f"{tag.replace(' ', '_').replace('/', '_')}_models.csv"
        output_path = os.path.join(output_dir, file_name)

        # Save the filtered DataFrame to a CSV file
        tag_df.to_csv(output_path, index=False)

    return f"Datasets saved in {output_dir}"



def group_and_count_by_tags(df, out_file):
    # Assuming tags are stored as a list in each row. If it's a string, additional splitting might be needed.
    df_exploded = df.explode('tag')
    grouped = df_exploded.groupby('tag')['model'].count().reset_index(name='count')
    grouped = grouped.sort_values(by='count', ascending=False)
    #print(grouped.shape)
    grouped.to_csv(out_file, index=False)
    return grouped

def sort_by_col(df,col, out):
    sorted_df = df.sort_values(by=col, ascending=False)
    sorted_df.to_csv(out, index=False)
    return sorted_df


def sort_by_likes(df,out):
    sorted_df = df.sort_values(by='likes', ascending=False)
    sorted_df.to_csv(out, index=False)
    return sorted_df

def sort_by_downloads(df, out):
    sorted_df = df.sort_values(by='Stars', ascending=False)
    sorted_df= sorted_df[:10]
    sorted_df.to_csv(out, index=False)
    return sorted_df

def sort_by_support(df,out):
    sorted_df = df.sort_values(by='count', ascending=False)
    sorted_df.to_csv(out, index=False)


def filter_unpopular(df, threshold, out):
    filtered_df = df[df['Usage Count'] >= threshold]
    #filtered_df.to_csv(out_file, index=False)
    filtered_df.to_csv(out, index=False)
    return


def is_yaml_well_formatted(input_df, yaml_column_name):
    """
    Check if the YAML content in the specified column is well-formatted.

    Parameters:
    - input_df: DataFrame containing the column to check.
    - yaml_column_name: Name of the column containing YAML content.

    Returns:
    - A boolean Pandas Series indicating whether each row's YAML content is well-formatted.
    """

    def check_format(yaml_string):
        try:
            yaml.safe_load(yaml_string)
            return True
        except yaml.YAMLError:
            return False

    return input_df[yaml_column_name].apply(check_format)


def parse_yaml_and_update_df(input_df, yaml_column_name, field_to_remove=None, new_column_name='preprocessed_yaml'):
    """
    Parse and preprocess YAML content within a DataFrame column, updating the DataFrame
    to include this preprocessed content in a new column.

    Parameters:
    - input_df: DataFrame containing the YAML content.
    - yaml_column_name: Name of the column with YAML strings.
    - field_to_remove: Optional field name to remove from the YAML data.
    - new_column_name: Name for the new column to store preprocessed YAML data.

    Returns:
    - Updated DataFrame with the original data plus a new column with preprocessed YAML content.
    """

    def remove_field_from_data(data, field):
        if field:
            if isinstance(data, dict) and field in data:
                del data[field]
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and field in item:
                        del item[field]
        return data

    def normalize_data(data):
        # This function now returns data as-is, since we're directly adding it to a DataFrame column
        return data

    # Preprocess YAML content, including any field removal or normalization
    preprocessed_data = input_df[yaml_column_name].apply(lambda x: yaml.safe_load(x)).apply(
        lambda x: remove_field_from_data(x, field_to_remove)).apply(normalize_data)

    # Add the preprocessed data as a new column to the original DataFrame
    input_df[new_column_name] = preprocessed_data.apply(lambda x: yaml.dump(x))

    return input_df








def parse_yaml_to_df(input_df, yaml_column_name, field_to_remove=None):
        """
        Parse YAML content to a DataFrame and optionally remove a specified field.
        Handles cases where parsed YAML is a dictionary or a list.

        Parameters:
        - input_df: DataFrame containing the YAML content.
        - yaml_column_name: Name of the column with YAML strings.
        - field_to_remove: Optional field name to remove from the YAML data.

        Returns:
        - A DataFrame with the parsed and optionally modified YAML content.
        """

        def remove_field_from_data(data, field):
            if field:
                if isinstance(data, dict) and field in data:
                    del data[field]
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and field in item:
                            del item[field]
            return data

        def normalize_data(data):
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                # Here, you need to decide how to handle lists. As an example, we're converting each item into a row.
                # This may require adjusting to fit your specific data structure and needs.
                return {f'item_{i}': item for i, item in enumerate(data)}
            else:
                return {}

        parsed_data = input_df[yaml_column_name].apply(yaml.safe_load).apply(
            lambda x: remove_field_from_data(x, field_to_remove)).apply(normalize_data)

        output_df = pd.json_normalize(parsed_data)

        output_df.to_csv('preprocessed_dataset_stats.csv', index=False)
        return output_df


def compute_mean_median(dataframe, column_name):
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    mean_value = dataframe[column_name].mean()
    median_value = dataframe[column_name].median()



    return mean_value, median_value


def filter_and_drop_infrequent(df_main, df_check, threshold_freq, threshold_downloads, out_file):

    # Find elements in df_check with frequency >= threshold
    frequent_elements = df_check[df_check['count'] >= threshold_freq]['tags']

    # Filter df_main to keep rows where element is in the list of frequent elements
    filtered_df = df_main[df_main['tags'].isin(frequent_elements)]
    filter_unpopular(filtered_df, threshold_downloads)
    filtered_df.to_csv(out_file, index=False)
    return filtered_df


def map_models_to_github(df, mapped, not_mapped):

    with open(mapped, 'w', encoding='utf-8', errors='ignore') as map_file:
        with open(not_mapped, 'w', encoding='utf-8', errors='ignore') as missed:
            g = Github(cf.TOKEN)

            #model_to_repo_map = {}
            print('searching github')

            for model_name in df['model_name'].values.astype(str):
                # Extract the GitHub repository owner and repo name
                info_model = str(model_name).split('/')
                try:
                    info_model.remove('models')
                    repo_model = '/'.join(info_model)
                    query = f" repo:{repo_model}"


                    # Search for repositories matching this name
                    repos = g.search_repositories(query)

                    for repo in repos:
                        # Check Levenshtein distance
                        if levenshtein.distance(repo.name.lower(), repo_model.lower()) < 3:
                            # Store or process the matched repository
                            #model_to_repo_map[model_name] = repo.full_name
                            print("matched", model_name)
                            map_file.write(f'{repo.full_name}\n')
                        else:
                            missed.write(f'{repo.full_name}\n')
                except:
                    print("skipping", model_name)

    return






def write_tuples_to_csv(file_name, tuple_list, header):
    df = pd.DataFrame(tuple_list,columns=header)

    df.dropna(axis='index', how='any', inplace=True)
    df = df[df['card_data'] != 'None']
    df = df[df['tags'] != 'None']

    df.to_csv(file_name, index=False)
    return df


def remove_special_characters(input_string):
    if input_string:
        pattern = re.compile(r'[^\w\s]', re.UNICODE)
        return pattern.sub('', input_string)

def remove_rows(df, column):
    return df[df[column].isna()]


