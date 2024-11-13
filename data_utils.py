import pandas as pd
import re
import config as cf
from github import Github
import requests
from textdistance import levenshtein
import yaml
import os


<<<<<<< HEAD
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

=======
>>>>>>> 57bf30b63085dad699d6150f2a377862df51d7ce

def group_and_count_by_tags(df, out_file):
    # Assuming tags are stored as a list in each row. If it's a string, additional splitting might be needed.
    df_exploded = df.explode('tags')
    grouped = df_exploded.groupby('tags')['model_name'].count().reset_index(name='count')
    grouped = grouped.sort_values(by='count', ascending=False)
    #print(grouped.shape)
    grouped.to_csv(out_file, index=False)
    return grouped

<<<<<<< HEAD
def sort_by_likes(df,out):
    sorted_df = df.sort_values(by='likes', ascending=False)
    sorted_df.to_csv(out, index=False)
    return sorted_df

def sort_by_downloads(df, out):
    sorted_df = df.sort_values(by='downloads', ascending=False)
    sorted_df.to_csv(out, index=False)
    return sorted_df

def sort_by_support(df,out):
    sorted_df = df.sort_values(by='count', ascending=False)
    sorted_df.to_csv(out, index=False)
=======
def sort_by_likes(df):
    sorted_df = df.sort_values(by='likes', ascending=False)
    sorted_df.to_csv('stats/ranked_by_likes.csv', index=False)
    return sorted_df

def sort_by_downloads(df):
    sorted_df = df.sort_values(by='downloads', ascending=False)
    sorted_df.to_csv('stats/ranked_by_downloads.csv', index=False)
    return sorted_df

def sort_by_support(df):
    sorted_df = df.sort_values(by='count', ascending=False)
    sorted_df.to_csv('stats/sorted_by_freq.csv', index=False)
>>>>>>> 57bf30b63085dad699d6150f2a377862df51d7ce
    return sorted_df

def filter_unpopular(df, threshold):
    filtered_df = df[df['downloads'] >= threshold]
    #filtered_df.to_csv(out_file, index=False)
    return filtered_df


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
<<<<<<< HEAD
    df.dropna(axis='index', how='any', inplace=True)
    df = df[df['card_data'] != 'None']
    df = df[df['tags'] != 'None']
=======
    #df.dropna(axis='index', how='any', inplace=True)
    #df = df[df['card_data'] != 'None']
    #df = df[df['tags'] != 'None']
>>>>>>> 57bf30b63085dad699d6150f2a377862df51d7ce
    df.to_csv(file_name, index=False)
    return df


def remove_special_characters(input_string):
    if input_string:
        pattern = re.compile(r'[^\w\s]', re.UNICODE)
        return pattern.sub('', input_string)

def remove_rows(df, column):
    return df[df[column].isna()]


