import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import dropbox
import csv

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import tensorflow as tf
from tensorflow.keras import models, layers
from ast import literal_eval

t = datetime(2019, 1, 1)

st.set_page_config(layout = 'wide')

d = 1

# 原始資料處理；衍生欄位製作
@st.cache_data(max_entries = 1)
def read_raw_data(path):
  raw_data_path = path
  _, dbx_file = dbx.files_download(raw_data_path)
  csv_reader = csv.reader(dbx_file.content.decode('utf-8-sig').splitlines(), delimiter = ',')
  ls = []
  for row in csv_reader:
      ls.append(row)
  raw_data = pd.DataFrame(ls[1:], columns = ls[0])

  #把日期相關欄位轉為日期格式 yyyy-mm-dd hh:mm:ss
  raw_data['order_time'] = pd.DatetimeIndex(raw_data['order_time'])
  raw_data['first_order'] = pd.DatetimeIndex(raw_data['first_order'])
  raw_data['concert_time'] = pd.DatetimeIndex(raw_data['concert_time'])

  raw_data['itemID'] = raw_data['itemID'].apply(lambda x : int(x))
  raw_data['userID'] = raw_data['userID'].apply(lambda x : int(x))


  # 衍生欄位
  raw_data['dow'] = raw_data['concert_time'].dt.day_name()
  # 使用者下訂時間距音樂會時間分組
  # 當天: 0；1~7天: 1；8~14天: 2；15~30天: 3；31~60天: 4；60天以上: 5；沒買: 6
  raw_data['order_concert'] = raw_data['concert_time'] - raw_data['order_time']
  raw_data['order_concert'] = raw_data['order_concert'].dt.days
  raw_data['order_concert_group'] = raw_data['order_concert'].apply(lambda x : 0 if x == 0 else
                                                                    1 if x <= 7 else
                                                                    2 if x <= 14 else
                                                                    3 if x <= 30 else
                                                                    4 if x <= 60 else 5)

  return raw_data

@st.cache_data
def train_filter(train_set):
  while True:
    item_group = train_set.groupby('itemID').size().to_frame('item_count').reset_index()
    item_group = item_group[item_group['item_count'] > 1]
    train_set = train_set[train_set['itemID'].isin(list(item_group['itemID']))]

    user_group = train_set.groupby('userID').size().to_frame('user_count').reset_index()
    user_group = user_group[user_group['user_count'] > 1]
    train_set = train_set[train_set['userID'].isin(list(user_group['userID']))]

    if min(item_group['item_count']) > 1 and min(user_group['user_count']) > 1:
      return train_set  

@st.cache_data
def train_expand(d):
  train_time = t + relativedelta(days = d*15)
  train_set = raw_data[raw_data['order_time'] < train_time]

  # 只保留有一筆以上的資料
  # 大約會少掉 20,000筆
  train_set = train_filter(train_set)

  # item-itme model
  #user-item matrix、物品相似度
  user_item, item_sim = calculate_similarity(train_set)

  # 使用者最近一次下單 (recent)
  recent = train_set[['userID', 'order_time']].sort_values(by = 'order_time', ascending = False).drop_duplicates(subset='userID')
  recent['order_time'] = (train_time - recent['order_time']).dt.days
  recent['recent_group'] = recent['order_time'].apply(lambda x : 0 if x <=7 else
                                                      (1 if x <= 14 else
                                                       2 if x <= 21 else 
                                                       3 if x <= 31 else
                                                       4 if x <= 60 else
                                                       5 if x <= 90 else
                                                       6 if x <= 180 else
                                                       7 if x <= 365 else
                                                       8 if x <= 730 else
                                                       9 if x <= 1095 else
                                                       10))
  recent_group_dict = recent[['userID', 'recent_group']].set_index('userID').T.to_dict('list')

  train_set = train_set.drop_duplicates(subset = ['userID', 'itemID'])

  # 訓練集中 train_time 後尚在架上的節目
  avail_event = train_set[(train_set['concert_time'] > train_time) & (train_set['first_order'] <= train_time)]['itemID'].unique()
  avail_user = train_set['userID'].unique()
  
  feature_dict = {'userID' : train_set['userID'].unique(),
                  'itemID' : train_set['itemID'].unique(),
                  'dow' : train_set['dow'].unique(),
                  'inter' : train_set['inter'].unique(),
                  'category' : train_set['category'].unique(),
                  'recent_group' : recent['recent_group'].unique(),
                  'order_concert_group' : np.append(train_set['order_concert_group'].unique(), [6])} # expand 的類別要加進去

  user_id_idx, event_id_idx = make_idx(train_set)
  
  return user_item, item_sim, train_time, avail_event, avail_user, user_id_idx, event_id_idx, recent_group_dict, feature_dict

@st.cache_data(max_entries = 1)
def make_idx(train_data):
  # 音樂會編流水號
  events = train_data['itemID'].drop_duplicates().sort_values().to_frame()
  events['item_idx'] = range(0, len(events))
  event_id_idx = events.set_index('itemID').T.to_dict('list')

  # 使用者編流水號
  users = train_data['userID'].drop_duplicates().sort_values().to_frame()
  users['user_idx'] = range(0, len(users))
  user_id_idx = users.set_index('userID').T.to_dict('list')

  return user_id_idx, event_id_idx

@st.cache_data(max_entries = 1)
def create_test(raw_data, train_time):
  test = raw_data[raw_data['order_time'] >= train_time]
  test = test[(test['userID'].isin(avail_user)) & (test['itemID'].isin(avail_event))]
  test = test.drop_duplicates(subset = ['userID', 'itemID'])
  # itemID 於 train 內的 idx
  #test['user_idx'] = test['userID'].apply(lambda x : user_id_idx[x][0]) #要做為節目推薦使用者時可能需要
  test['item_idx'] = test['itemID'].apply(lambda x : item_id_idx[x][0])

  # test 內 user 編流水號
  # 為了網頁輸入
  test_user = test['userID'].drop_duplicates().sort_values().to_frame()
  test_user['test_idx'] = [n for n in range(len(test_user))]
  test_user_idx = test_user.set_index('test_idx').T.to_dict('list')
  return test, test_user_idx

@st.cache_data(max_entries = 1)
def title_enc_file(path):
    title_data_path = path
    _, dbx_file = dbx.files_download(title_data_path)
    csv_reader = csv.reader(dbx_file.content.decode('utf-8-sig').splitlines(), delimiter = ',')
    ls = []
    for row in csv_reader:
        ls.append(row)
    title = pd.DataFrame(ls[1:], columns = ls[0])
    title['itemID'] = title['itemID'].apply(lambda x : int(x))
    return title

@st.cache_resource(max_entries = 1)
def create_model(dw_path, w_path):
  users_in = layers.Input(name="users_in", shape=(1,))
  items_in = layers.Input(name="items_in", shape=(1,))
  dows_in = layers.Input(name='dows_in', shape=(1,), dtype=tf.string)
  inter_in = layers.Input(name='inter_in', shape=(1,), dtype=tf.string)
  cat_in = layers.Input(name='cat_in', shape=(1,), dtype=tf.string)
  recent_group_in = layers.Input(name='recent_group_in', shape=(1,))
  order_concert_group_in = layers.Input(name='order_concert_group_in', shape=(1,))
  title_in = layers.Input(name='title_in', shape=(512,))
    
  user_embedding = int(len(feature_dict['userID'])/2)
  item_embedding = int(len(feature_dict['itemID'])/2)
  dow_embedding =  int(len(feature_dict['dow'])/2)
  inter_embedding = int(len(feature_dict['inter'])/2)
  cat_embedding = int(len(feature_dict['category'])/2)
  recent_embedding = int(len(feature_dict['recent_group'])/2)
  order_embedding = int(len(feature_dict['order_concert_group'])/2)
  cf_emb = min(user_embedding, item_embedding)

  drop_out = 0.2
  l2 = 0.02

  ##  ===================== ##
  # feature embedding
  users_emb = layers.Embedding(name='users_emb', input_dim=len(feature_dict['userID'])+1,
                               output_dim=user_embedding)(users_in)
  users_emb = layers.Reshape(target_shape=(user_embedding,))(users_emb)

  items_emb = layers.Embedding(name='items_emb', input_dim=len(feature_dict['itemID'])+1,
                               output_dim=item_embedding)(items_in)
  items_emb = layers.Reshape(target_shape=(item_embedding, ))(items_emb)

  # cf embedding
  users_emb_cf = layers.Embedding(name="users_emb_cf", input_dim=len(feature_dict['userID'])+1,
                               output_dim=cf_emb)(users_in)
  users_emb_cf = layers.Reshape(target_shape = (cf_emb,))(users_emb_cf)

  items_emb_cf = layers.Embedding(name="items_emb_cf", input_dim=len(feature_dict['itemID'])+1,
                               output_dim=cf_emb)(items_in)
  items_emb_cf = layers.Reshape(target_shape = (cf_emb,))(items_emb_cf)

  ## dow embedding
  dows_sl= layers.StringLookup(vocabulary=feature_dict['dow'], mask_token=None, name='dow_sl')
  dows_emb = dows_sl(dows_in)
  dows_emb = layers.Embedding(name='dows_emb', input_dim=len(feature_dict['dow'])+1,
                              output_dim=dow_embedding)(dows_emb)
  dows_emb = layers.Reshape(target_shape = (dow_embedding,))(dows_emb)

  ## inter embedding
  inters_sl= layers.StringLookup(vocabulary=feature_dict['inter'], name='inters_sl')
  inters_emb = inters_sl(inter_in)
  inters_emb = layers.Embedding(name='inters_emb', input_dim=len(feature_dict['inter'])+1,
                                output_dim=inter_embedding)(inters_emb)
  inters_emb = layers.Reshape(target_shape = (inter_embedding,))(inters_emb)

  ## category embedding
  cats_sl= layers.StringLookup(vocabulary=feature_dict['category'], name='cats_sl')
  cats_emb = cats_sl(cat_in)
  cats_emb = layers.Embedding(name='cats_emb', input_dim=len(feature_dict['category'])+1,
                              output_dim=cat_embedding)(cats_emb)
  cats_emb = layers.Reshape(target_shape = (cat_embedding,))(cats_emb)

  ## recent group
  recent_il = layers.IntegerLookup(vocabulary=feature_dict['recent_group'])
  recent_emb = recent_il(recent_group_in)
  recent_emb = layers.Embedding(name='recent_emb', input_dim=len(feature_dict['recent_group'])+1,
                               output_dim=recent_embedding)(recent_emb)
  recent_emb = layers.Reshape(target_shape = (recent_embedding,))(recent_emb)

  ## order_concert_group
  order_il = layers.IntegerLookup(vocabulary=feature_dict['order_concert_group'])
  order_emb = order_il(order_concert_group_in)
  order_emb = layers.Embedding(name='order_emb', input_dim=len(feature_dict['order_concert_group'])+1,
                               output_dim=order_embedding)(order_emb)
  order_emb = layers.Reshape(target_shape = (order_embedding,))(order_emb)

  # title_enc
  title_emb = layers.Dense(512)(title_in)

  ##  ===================== ##
  # cf
  cf_xx = tf.math.multiply(users_emb_cf, items_emb_cf)

  ##  ===================== ##
  # nn
  feature_embs = tf.concat([users_emb, items_emb, order_emb,
                            cats_emb, dows_emb,
                            recent_emb, inters_emb,
                            title_emb], axis = 1)

  nn_layer = layers.Dense(256, activation = 'relu',
                          kernel_regularizer=tf.keras.regularizers.L2(l2))(feature_embs)
  nn_layer = layers.Dropout(drop_out)(nn_layer)
  nn_layer = layers.Dense(128, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.L2(l2))(nn_layer)
  nn_layer = layers.Dropout(drop_out)(nn_layer)
  nn_layer = layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.L2(l2))(nn_layer)
  nn_layer = layers.Dense(16, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.L2(l2))(nn_layer)

  ##  ===================== ##
  # concat everything
  final_nn = tf.concat([cf_xx, nn_layer], axis = 1)

  y_out = layers.Dense(units=2, activation = 'softmax', name='y_out')(final_nn)

  model = models.Model(inputs=[users_in, items_in,
                               dows_in,  inter_in, cat_in, recent_group_in,
                               order_concert_group_in, title_in],
                      outputs=y_out, name="Neural_CollaborativeFiltering")
  
  dbx.files_download_to_file(dw_path, w_path)
  model.load_weights(dw_path)
  os.remove(dw_path)
 
  return model

@st.cache_data(max_entries = 1)
def calculate_similarity(train):
  train['rating'] = 1
  user_item = train.pivot_table(values='rating', #先只看有或沒有，unary
                               index='userID', 
                               columns='itemID',
                               fill_value=0)
  
  data_sparse = sparse.csr_matrix(user_item)
  #原本列為使用者、欄為item，轉置後列(x)為item，計算item-item的cosine similarity
  similarities = cosine_similarity(data_sparse.transpose())
  sim = pd.DataFrame(data=similarities, index= user_item.columns, columns= user_item.columns)
  
  return user_item, sim

#%% Dropbox setting; download raw_data & title
# dropbox
@st.cache_data
def log_in(app_key, app_secret, oauth2_refresh_token):
    return dropbox.Dropbox(app_key = app_key,
            app_secret = app_secret,
            oauth2_refresh_token = oauth2_refresh_token)

app_key = st.secrets.dbx_credentials.app_key
app_secret = st.secrets.dbx_credentials.app_secret
oauth2_refresh_token = st.secrets.dbx_credentials.oauth2_refresh_token

dbx = log_in(app_key, app_secret, oauth2_refresh_token)

# 產生raw_data
# userID', 'order_time', 'itemID', 'title', 'concert_time','first_order', 'inter', 'category', 'dow'
raw_data = read_raw_data('/data_230729.csv')

title = title_enc_file('/title_vec.csv')


#%% prediction

@st.cache_data(max_entries = 1)
def predict_func(user):    
    user_id = test_user_idx[user][0] #輸入為 test 內的流水編號，查會員編號
    user_idx = user_id_idx[user_id][0] #會員編號查 train 內的流水編號
    
    ##========== item-item ==========##
    item_sim_sum = item_sim.sum(axis = 1) #分母
    score_item = item_sim.dot(user_item.loc[user_id].transpose()) #分子
    score_item = pd.DataFrame(score_item.transpose().values / item_sim_sum.values, index = score_item.transpose().index, columns=['score'])

    #計算 avail_event 的可能性分數
    trad_avail_event = score_item.loc[avail_event].reset_index()
    trad_avail_event = trad_avail_event.merge(raw_data[['itemID', 'title', 'inter', 'category', 'dow']], on = 'itemID', how = 'left').drop_duplicates(subset = 'itemID')
    trad_avail_event = trad_avail_event.sort_values(by = 'score', ascending = False)
    trad_avail_event = trad_avail_event.drop_duplicates(subset = 'title')
    trad_avail_event['No.'] = [n for n in range(1, len(trad_avail_event)+1)]
    trad_avail_event = trad_avail_event[['No.', 'title', 'inter', 'category', 'dow', 'score']].reset_index(drop = True)
    trad_avail_event['score'] = trad_avail_event['score'].map('{:.2f}'.format)
    trad_rec_items = trad_avail_event[:10][['No.', 'title']]
    

    ##========== NCF ==========##
    # 需要重建 item 資料
    rec_items = pd.DataFrame(avail_event, columns = ['itemID'])
    rec_items = rec_items.merge(raw_data[['itemID', 'title', 'concert_time', 'dow', 'inter', 'category', 'order_concert_group']],
                                on ='itemID', how = 'left').drop_duplicates(subset = 'itemID')
    rec_items['item_idx'] = rec_items['itemID'].apply(lambda x : item_id_idx[x][0])

    # user_idx 併到每一個 rec_items
    rec_items['user_idx'] = user_idx

    # 回查 userID
    # 用 userID 查 recent_group
    rec_items['userID'] = user_id
    rec_items['recent_group'] = rec_items['userID'].apply(lambda x : recent_group_dict[x][0])

    # 找出 test_enc   
    rec_items = rec_items.merge(title[['itemID', 'list']], how = 'left', on = ['itemID'])
    title_enc = rec_items['list']
    title_enc = title_enc.apply(lambda x : literal_eval(x))
    title_enc = np.stack(title_enc.values)

    preds = model.predict([rec_items['user_idx'], rec_items['item_idx'], rec_items['dow'],
                           rec_items['inter'], rec_items['category'], rec_items['recent_group'],
                           rec_items['order_concert_group'],
                           title_enc], verbose = 0)
    
    preds = pd.DataFrame(preds, columns = ['pred_0', 'pred_1'])
    rec_items = rec_items.reset_index(drop = True)
    rec_items = rec_items.merge(preds['pred_1'].rename('pred'), left_index = True, right_index = True)
    rec_items = rec_items.sort_values(by = 'pred', ascending = False).reset_index(drop = True)
    rec_items = rec_items.drop_duplicates(subset = 'title')
    
    # 使用者訓練集中購買節目
    train_event = raw_data[raw_data['order_time'] < train_time]
    train_event = train_filter(train_event)
    past_event = train_event[train_event['userID'] == user_id][['itemID', 'title']].drop_duplicates().reset_index(drop = True)
    train_event = train_event.groupby('itemID')['userID'].count().reset_index()
    past_event = past_event.merge(raw_data[['itemID', 'inter', 'category', 'dow']], 'left', on = 'itemID').drop_duplicates()
    past_event = past_event.merge(train_event[['itemID', 'userID']], 'left', on = 'itemID')
    past_event['No.'] = [n for n in range(1, len(past_event) + 1)]
    past_event = past_event[['No.', 'title', 'inter', 'category', 'dow', 'userID']].rename(columns = {'userID' : '訓練集中總交易筆數'})

    # 將使用者以買過的節目自推薦清單中移除
    rec_items = rec_items[~rec_items['title'].isin(past_event['title'])].reset_index(drop = True)
    rec_items['rank'] = rec_items.index + 1
    
    # 候選節目
    avail_event_df = pd.DataFrame(avail_event, columns = ['itemID'])
    avail_event_df = avail_event_df.merge(raw_data[['itemID', 'title', 'inter', 'category', 'dow']], 'left', on = 'itemID')
    avail_event_df = avail_event_df.merge(rec_items[['itemID', 'pred']], 'left', on = 'itemID')
    avail_event_df = avail_event_df.sort_values(by = 'pred', ascending = False)
    avail_event_df = avail_event_df.drop_duplicates(subset = 'title').reset_index(drop = True)
    avail_event_df['No.'] = [n for n in range(1, len(avail_event_df) + 1)]
    avail_event_df = avail_event_df[['No.', 'title', 'inter', 'category', 'dow', 'pred']]
    avail_event_df['pred'] = avail_event_df['pred'].map('{:.2%}'.format)
    
    # 推薦清單減為10
    rec_items = rec_items.iloc[:10][['rank', 'title']]
    
    # 使用者實際購買節目
    actual_event = test[test['userID'] == user_id].drop_duplicates(subset='title')
    actual_event['No.'] = [n for n in range(1, len(actual_event) + 1)]
    actual_event = actual_event[['No.', 'title', 'inter', 'category', 'dow']]

    # Hit 列上色
    # actual_event 於 availe_event_df 中的 index
    hit = list(avail_event_df[avail_event_df['title'].isin(actual_event['title'])].index)
    trad_hit = list(trad_avail_event[trad_avail_event['title'].isin(actual_event['title'])].index)
    
    actual_event['傳統 CF 推薦序'] = [n + 1 for n in trad_hit]
    actual_event['NCF 推薦序'] = [n + 1 for n in hit]

    # 表格標題
    rec_items = rec_items.rename(columns = {'rank' : '推薦序', 'title' : '音樂會標題'})
    trad_rec_items = trad_rec_items.rename(columns = {'No.' : '推薦序', 'title' : '音樂會標題'})
    avail_event_df = avail_event_df.rename(columns = {'title' : '音樂會標題', 'pred' : '購票機率'})
    trad_avail_event = trad_avail_event.rename(columns = {'title' : '音樂會標題', 'score' : '購票可能性'})
    actual_event = actual_event.rename(columns = {'title' : '音樂會標題'})
    past_event = past_event.rename(columns = {'title' : '音樂會標題'})
    
    return trad_rec_items, trad_avail_event, rec_items, avail_event_df, actual_event, past_event, hit, trad_hit

#%% streamlit

@st.cache_data(max_entries = 1)
def row_bgc(source):
    color = 'background-color : {}'.format
    ls = []
    for n in range(source.shape[0]):
        if n in hit:
            ls.append(1)
        else: ls.append(0)
    mask = pd.DataFrame([ls]*source.shape[1]).T
    style = np.where(mask, color('#d5edfa'), color('white'))
    return style

@st.cache_data(max_entries = 1)
def trad_row_bgc(source):
    color = 'background-color : {}'.format
    ls = []
    for n in range(source.shape[0]):
        if n in trad_hit:
            ls.append(1)
        else: ls.append(0)
    mask = pd.DataFrame([ls]*source.shape[1]).T
    style = np.where(mask, color('#d5edfa'), color('white'))
    return style

st.header('訓練時間：2019/01/16')

# 清除 cache 資料
if 'd' not in st.session_state:
    st.session_state['d'] = d

if st.session_state['d'] != d:
    st.cache_resource.clear()
    calculate_similarity.clear()
    train_filter.clear()
    train_expand.clear()
    make_idx.clear()
    create_test.clear()
    st.session_state['d'] = d
    
# 產生基本資料
user_item, item_sim, train_time, avail_event, avail_user, user_id_idx, item_id_idx, recent_group_dict, feature_dict = train_expand(d)
test, test_user_idx = create_test(raw_data, train_time)

# re-create model
# 建立模型、讀入權重
# 下載檔案、讀入後刪除
w_path = '/model_' + str(d) + '_weights.h5'
dw_path = 'model_' + str(d) + '_weights.h5'
model = create_model(dw_path, w_path)


member_range = '輸入使用者代碼：　（從 0 到 ' + str(len(test_user_idx)-1) + ' 的整數）'
st.write('**兩個模型表現都很好**：使用者代碼 44')

user_n = st.number_input(member_range, min_value = 0, max_value = len(test_user_idx)-1)

if st.button('產生推薦'):
    trad_rec_items, trad_avail_event, rec_items, avail_event_df, actual_event, past_event, hit, trad_hit = predict_func(user_n)
    
    st.markdown('''
              <style>
              .define_size{
                  font-size : 16pt}
              </style>
              ''', unsafe_allow_html = True)
        
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p class = 'define_size'>傳統 CF 模型推薦節目</p>", unsafe_allow_html = True)
            st.dataframe(trad_rec_items.style.apply(trad_row_bgc, axis = None),
                         hide_index = True)
        with col2:
            st.markdown("<p class = 'define_size'>NCF模型推薦節目</p>", unsafe_allow_html = True)
            st.dataframe(rec_items.style.apply(row_bgc, axis = None),
                         hide_index = True)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p class = 'define_size'>傳統 CF 對販售中節目預測分數</p>", unsafe_allow_html = True)
            st.dataframe(trad_avail_event.style.apply(trad_row_bgc, axis = None),
                         hide_index = True)
        with col2:
            st.markdown("<p class = 'define_size'>NCF 對販售中節目預測機率</p>", unsafe_allow_html = True)
            st.dataframe(avail_event_df.style.apply(row_bgc, axis = None),
                         hide_index = True)

        
    with st.container():
        st.markdown("<p class = 'define_size'>使用者實際購買節目</p>", unsafe_allow_html = True)
        st.dataframe(actual_event, hide_index = True)
    
    with st.container():
        st.markdown("<p class = 'define_size'>使用者過去曾購買節目<p>", unsafe_allow_html= True)
        st.dataframe(past_event, hide_index = True)
