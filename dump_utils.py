
import mysql.connector
from mysql.connector import Error
import data_utils as du


def create_server_connection(host_name, user_name, user_password, db_name, port_number):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name,
            port=port_number
        )
        print("MySQL Database connection successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


def get_tag_by_model(conn, model_name):
    list_results = []
    cur = conn.cursor()

    model_name = "models/" + model_name

    cur.execute("SELECT model_id, pipeline_tag"
                " FROM model"
                " WHERE model_id = %s;",  (model_name,))

    data = cur.fetchall()



    return data


def get_dataset_description(conn):
    dict_datasets = {}
    cur = conn.cursor()
    cur.execute("SELECT id, description FROM dataset,repository where dataset.dataset_id=repository.id;")
    data = cur.fetchall()
    dict_datasets.update({data[0]: data[1]})
    return dict_datasets

def get_model_data(conn):
    list_results = []
    cur = conn.cursor()

    cur.execute("SELECT model.model_id,repository.card_data,model.pipeline_tag,model.likes,downloads"
                " FROM model,repository"
                " where model.model_id = repository.id;")




    data = cur.fetchall()
    for d in data:
        cleaned_tuple = [str(elem).strip().replace(',','') for elem in d]
        list_results.append(cleaned_tuple)


    #list_results.append(cur.fetchall())
    headers = ['model_name', 'card_data', 'tags', 'likes', 'downloads']


    du.write_tuples_to_csv('datasets/card_and_tag_dump_june.csv', list_results, headers)

    return list_results


def get_class_models(conn):
    list_results = []
    cur = conn.cursor()
    text = "text-classification"
    image = "image-classification"
    token = "token-classification"
    tab = "tabular-classification"
    cur.execute("SELECT model.model_id,repository.card_data,model.pipeline_tag,model.likes,downloads "
                "FROM model,repository where model.model_id = repository.id"
                " AND (model.pipeline_tag = 'text-classification' OR model.pipeline_tag = 'image-classification'"
                " OR model.pipeline_tag = 'token-classification' OR  model.pipeline_tag = 'tabular-classification' );")

    data = cur.fetchall()
    for d in data:
        cleaned_tuple = [str(elem).strip().replace(',','') for elem in d]
        list_results.append(cleaned_tuple)


    #list_results.append(cur.fetchall())
    headers = ['model_name', 'card_data', 'tags', 'likes', 'downloads']

    du.write_tuples_to_csv('datasets/ptm_fair.csv', list_results, headers)



    return list_results