#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[42]:


pip install sqlalchemy


# In[2]:


from sqlalchemy import create_engine


# In[3]:


import sqlalchemy


# In[47]:


pymysql.install_as_MySQLdb()


# In[4]:


import MySQLdb


# In[3]:


# connection 정보
engine = pymysql.connect(
    host = 'localhost', # host name
    port = 3306,
    user = 'root', # user name
    password = '7806', # password
    db = 'test_workbench', # db name
)


# In[36]:


conn = engine.connect()


# In[37]:


data = pd.read_csv('C:/Users/aiicon-KJG/Desktop/total_data_13_17.csv',encoding='cp949')


# In[38]:


data = data[:100]


# In[4]:


from dateutil.parser import parse


# In[83]:


temp = data['AUCNG_DE']


# In[49]:


engine = create_engine("mysql+mysqldb://ai_master:ai_master@211.248.97.155:3306/ai_db", encoding='utf-8')


# In[51]:


conn = engine.connect()


# In[84]:


dtype = {
    'AUCNG_DE':sqlalchemy.VARCHAR(9)
}


# In[85]:


temp.to_sql(name='test1', con=engine, if_exists='append', index=False, dtype = dtype)


# In[ ]:





# In[16]:


pip install pymysql


# In[5]:


import pymysql
import pymysql.cursors
import mysql.connector
import pandas as pd


# In[11]:


pip install mysql-connector-python


# In[7]:


# connection 정보
conn = pymysql.connect(
    host = 'localhost', # host name
    port = 3306,
    user = 'root', # user name
    password = '7806', # password
    db = 'NIX_tech', # db name
    charset = 'utf8'
)


# In[9]:


# 테이블 생성 sql 정의
sql = '''
CREATE TABLE AEO_Public_Company (
    No INT not null,
    Com_name VARCHAR(20) not null,
    com_rank varchar(10) not null
) ENGINE=InnoDB DEFAULT CHARSET=utf8
'''

# 테이블 생성
with conn.cursor() as cursor:
    cursor.execute(sql) # execute method를 호출하여 SQL명령 수행


# In[4]:


mydb = mysql.connector.connect(
    host = "localhost",
    user = "root", 
    password = "7806",
    database = "test_workbench"
)

mycursor = mydb.cursor() # cursor 객체 생성

sql = "INSERT INTO tb_student (name, address) VALUES (%s, %s)"
val = [
    ('CHOI', 'Lowstreet4'),
    ('YangCHOI', 'ulsan'),
    ('BAEK', 'busan'),
    ('PARK', 'seoul'),
    ('HAN', 'jinju')
]

mycursor.executemany(sql, val) # 다중 실행 함수

mydb.commit() # 트렌젝션의 변경내용을 db에 반영(입력 후 변화함수)

print(mycursor.rowcount, "record inserted")


# In[14]:


sql = "SELECT * FROM `tb_student`;"
mycursor.execute(sql)
result = mycursor.fetchall() # 	모든 데이터를 한 번에 가져올 때 사용, 배열형식으로 저장


# In[10]:


result = mycursor.fetchall()


# In[15]:


result_f = pd.DataFrame(result)
result_f


# In[ ]:





# In[ ]:





# In[38]:


df = pd.read_csv('19. 중소벤처기업진흥공단_중소기업 성과공유 기업 정보_20181231.csv' ,encoding='cp949')


# In[42]:


df


# In[69]:


df = df.drop(index=[400], axis=0)


# In[31]:


df = df.drop(columns = ['a'], axis=1)


# In[41]:


df.columns = ['class','com_name','com_manager','first_day','ex_day','ag_bonus','ag_to','ag_yo_to','ag_yo_wo','ag_mo','ag_wo','ag_in_wo','ag_sto','ag_job','int_bonus','int_to','int_yo_to','int_yo_wo','int_mo','int_wo','int_in_wo','int_sto','int_peo_re','int_job_dev','int_tal_pro','int_no_cul','int_yo_fr','int_fa_fr','com_contents','com_pro']


# In[25]:


df.info()


# In[24]:


df['cert_year'] = df['cert_year'].apply(lambda x: int(x))


# In[44]:


df['ex_day'] = df['ex_day'].apply(lambda x: pd.to_datetime(parse(str(x))))


# In[45]:


engine = create_engine("mysql+pymysql://ai_master:ai_master@211.248.97.155:3306/ai_db", encoding='utf-8')
conn = engine.connect()
df.to_sql(name = "nix_komes_result", con = engine, index = True)


# In[40]:


df.isnull().sum()


# In[34]:


df.fillna('N', inplace = True)


# In[96]:


df.dropna(inplace = True)


# In[ ]:




