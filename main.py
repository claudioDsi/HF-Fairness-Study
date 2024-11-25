import os

import dump_utils as du
import config as cf
from classifier import run_classifier
import pandas as pd

import mapping_utils as mp

import data_utils as d
import search_github_data as miner



def preprocessing_pipeline():

    df_dump = pd.read_csv('datasets/original_dump.csv')
    print(df_dump.shape)
    df_main = pd.read_csv(cf.INPUT_DATA_PATH)
    print(df_main.shape)
    df_tags = pd.read_csv(cf.SRC_TAG_FREQ)
    mean_d, median_d = d.compute_mean_median(df_main,'downloads')
    mean_f, median_f = d.compute_mean_median(df_tags, 'count')

    filtered_df = d.filter_and_drop_infrequent(df_main, df_tags, int(median_f), int(mean_d), 'datasets/d2.csv')
    print(filtered_df.shape)
    d0 = d.group_and_count_by_tags(df_dump, 'stats/d0_stats.csv')
    d1 = d.group_and_count_by_tags(df_main, 'stats/d1_stats.csv')
    d2 = d.group_and_count_by_tags(filtered_df, 'stats/d2_stats.csv')
    d0 = d.group_and_count_by_tags(df_dump, 'stats/d0_june_stats.csv')
    d1 = d.group_and_count_by_tags(df_main, 'stats/d1_june_stats.csv')
    d2 = d.group_and_count_by_tags(filtered_df, 'stats/d2_june_stats.csv')

def pipeline_fairness():
    df_all_class = pd.read_csv("datasets/ptm_fair.csv")
    d.group_and_count_by_tags(df_all_class, 'stats/models_class_stats.csv')
    d.sort_by_downloads(df_all_class, "stats/ptm_class_ranked.csv")
    mean, median = d.compute_mean_median(df_all_class,'downloads')
    d.filter_unpopular(df_all_class,mean,"datasets/popular_ptm_class.csv")
    d.split_df_by_tags(df_all_class, 'datasets/tagged_datasets/')



def collect_models_data():
    connection = du.create_server_connection(cf.HOST, cf.USER, cf.PWD, cf.DB, cf.PORT)
    #get_dataset_description(connection)
    du.get_model_data(connection)
    du.get_class_models(connection)
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")


def mapping_pipeline(ptm):
    similar_ptms = mp.search_ptm_name(ptm, 'datasets/d2.csv')
    #print(similar_ptms)
    print('Most frequent tag', mp.get_most_freq_pipeline_tag(similar_ptms)[0])
    macro, sub = mp.get_most_freq_se_task(ptm, cf.SRC_PAPERS, cf.SRC_SE_TASKS, cf.SRC_MACRO)

    print('Similar PTMs', similar_ptms)
    print('Macro SE task',set(macro))
    print('Sub-tasks', sub)
    return similar_ptms, macro, sub

def code_analysis():
    df = pd.read_csv('splitted_data/split_part_1.csv')

    models_name = df['model_name'].str.replace('models/', '', regex=False).tolist()
    token = "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"
    d.mine_github_source_code(models_name, token, "all_models_source_code_part_1.json")


if __name__ == '__main__':



    #df_origins = pd.read_csv("datasets/card_and_tag_dump_june.csv")
    #print(df_origins.shape)

    #df_class = pd.read_csv("datasets/ptm_fair.csv")
    #print(df_class.shape)

    # df_model_card = pd.read_csv("stats/ranked_by_likes_june.csv")
    # print(df_model_card.head(10))
    # df_fair = pd.read_csv("datasets/ptm_fair.csv")    #
    # d.sort_by_downloads(df_fair,"stats/ptm_fair.csv")
    #d.search_fairness_keywords(df=df_all,columns=['model_name','card_data'],output_path="datasets/filtered_ptms.csv")
    #
    # print(df_origins.shape)
    # print(df_model_card.shape)
    # print(df_all.shape)

    #d.sort_by_likes(df_dump)
    #d.sort_by_support(df_dump)

    #df_popular = pd.read_csv("stats/ranked_by_downloads_june.csv")
    # #print(df_popular.shape)

    #preprocessing_pipeline()


    #models_name=d.preprocessing_queries(models_name)
    #d.merge_json_files(json_file1="all_models_source_code.json", json_file2="all_models_source_code_2.json",output_file="all_models_source_code_merged.json")




    #d.search_keywords_and_libraries(json_file="all_models_source_code_merged.json",
                                   # output_csv="code_search_results.csv")


    #d.filter_models_with_matches_updated("code_search_results.csv","matched_model.csv")
    #print(df_results.describe())

    #df = pd.read_csv('Gao_work/sample_hf_mc.csv')


    code_analysis()

    #d.split_csv("datasets/ptm_fair.csv","splitted_data/",10000)



















