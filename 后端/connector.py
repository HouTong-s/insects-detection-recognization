import pymysql
import os
import random
import json
import enum


def connecting():
    return pymysql.connect(host='rm-uf60ufoc5i52mridp5o.mysql.rds.aliyuncs.com', port=3306, user='user1',
                           password='Abc@1234', db='insect_db')


def upload_temp(pic_name, annotation_js, temp_dic="temp"):
    connector = connecting()
    cursor = connector.cursor()
    print(annotation_js)
    annotation = json.loads(annotation_js)
    try:
        for region in annotation["regions"]:
            temp_tag = region["tags"]
            sql = 'insert into temp_table(pic_path,point1,point2, temp_order, temp_family, temp_genus, temp_species) ' \
                  'values(%s, %s, %s, %s,%s, %s, %s) '
            pic_path = temp_dic + "/" + pic_name[8:]
            point1 = json.dumps(region["points"][0])
            point2 = json.dumps(region["points"][2])
            temp_order = temp_tag["mu"]
            temp_family = temp_tag["ke"]
            temp_genus = temp_tag["shu"]
            temp_species = temp_tag["zhong"]
            values = (pic_path, point1, point2, temp_order, temp_family, temp_genus, temp_species)
            cursor.execute(sql, values)
    except Exception as e:
        print("something wrong")
        connector.rollback()
    finally:
        cursor.close()
        connector.commit()
        connector.close()


def log_in(user_name, password):
    connector = connecting()
    cursor = connector.cursor()
    sql = "SELECT password FROM admin_users WHERE user_id=='%s'"
    cursor.execute(sql, user_name)
    if password == cursor.fetchone()[0]:
        return True
    else:
        return False


def get_sample(sample_species: str, root="C:/Users/Administrator/Desktop/PestPictures", num=5):
    connector = connecting()
    cursor = connector.cursor()
    sql = "SELECT path FROM species WHERE "
    sql = sql + "species_name =%s"
    value = sample_species
    cursor.execute(sql, value)
    dic_name = cursor.fetchone()[0]

    sample_paths = []
    for file_name in os.listdir( root+ dic_name):
        if file_name.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            if random.randint(0, 3) % 4 == 0:
                sample_paths.append("http://139.224.50.124:8081/PestPictures"+dic_name + "/" + file_name)
                if len(sample_paths) >= num:
                    break
    cursor.close()
    connector.close()
    return sample_paths


def fetch_one_annotation():
    connector = connecting()
    cursor = connector.cursor()
    sql = "SELECT * FROM temp_table"
    cursor.execute(sql)
    path, mu, ke, shu, zhong, pt1, pt2 = cursor.fetchone()
    return path, mu, ke, shu, zhong, pt1, pt2


def delete_temp():
    return


def move_pic():
    return


def add_new(root, name, mu_name, ke_name, shu_name, dic_name):
    connector = connecting()
    cursor = connector.cursor()
    try:
        sql = "INSERT INTO species(order_name, genus_name, family_name, species_name, path) VALUES(%s, %s, %s, %s, %s)"
        values = (mu_name, shu_name, ke_name, name, dic_name)
        cursor.execute(sql, values)

    except Exception as e:
        print("something wrong")
        connector.rollback()
    finally:
        cursor.close()
        connector.commit()
        connector.close()


def fetch_specie(path):
    connector = connecting()
    cursor = connector.cursor()
    sql = "SELECT * FROM species WHERE path = %s"
    cursor.execute(sql, path)
    columns = cursor.fetchone()
    return columns


# add_new("", "shenme", "shenme", "shenme", "shenme", "/c")
# print(fetch_specie("/100"))
# add_new()
# p, _, _, _, _, _, p2 = fetch_one_annotation()
#
# print(p2)
# class MySqlConnector:
#     def __init__(self, pic_root_path=""):
#         self.root = pic_root_path
#
#     def rm_temp(self, objective_path):

# table_names = ['family', 'genus', 'order', 'species', 'temp_table', 'user']
# mysql_conn = pymysql.connect(host='rm-uf60ufoc5i52mridp5o.mysql.rds.aliyuncs.com', port=3306, user='user1',
#                              password='Abc@1234', db='insect_db')
# my_cursor = mysql_conn.cursor()
#
# sql = "SELECT * FROM temp_table"
#
# # for i in table_names:
# my_cursor.execute(sql)
# my_result = my_cursor.fetchall()
# print(my_result)

# js_sample = {"asset": {"format": "png", "size": {"width": 1534, "height": 512}}, "regions": [
#     {"tags": {"mu": "半翅目", "ke": "蝉科", "shu": "蚱蝉属", "zhong": "黑蚱蝉"}, "boundingBox": {"height": 159, "width": 595},
#      "points": [{"x": 940, "y": 354}, {"x": 1535, "y": 354}, {"x": 1535, "y": 513}, {"x": 940, "y": 513}]},
#     {"tags": {"mu": "半翅目", "ke": "兜蝽科", "shu": "皱蝽属", "zhong": "大皱蝽"}, "boundingBox": {"height": 215, "width": 758},
#      "points": [{"x": 72, "y": 172}, {"x": 830, "y": 172}, {"x": 830, "y": 387}, {"x": 72, "y": 387}]}]}
#
# js_sample = json.dumps(js_sample)
#
# upload_temp("picture.png", js_sample)

# get_sample(1, "鳞翅目", "大蚕蛾科", "尾大蚕蛾属", "红尾大蚕蛾")
