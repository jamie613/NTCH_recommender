import streamlit as st

st.set_page_config(layout = 'wide', initial_sidebar_state="expanded")
select_model = st.sidebar.radio('選擇頁面', ('main',
                                         'model_02',
                                         'model_12',
                                         'model_15',
                                         'model_16'))

st.markdown('''
            <style>
            .app {height : 80vh;
                  width : 100%}
            </style>''', unsafe_allow_html = True)


if select_model == 'model_02':
    html = '<iframe src="https://model-02.streamlit.app/?embed=true" class = "app"></iframe>'
    st.markdown(html , unsafe_allow_html = True)

elif select_model == 'model_12':
    html = '<iframe src="https://model-12.streamlit.app/?embed=true" class = "app"></iframe>'
    st.markdown(html, unsafe_allow_html = True)
        
elif select_model == 'model_15':
    html = '<iframe src="https://model-15.streamlit.app/?embed=true" class = "app"></iframe>'
    st.markdown(html, unsafe_allow_html = True)

elif select_model == 'model_16':
    html = '<iframe src="https://model-16.streamlit.app/?embed=true" class = "app"></iframe>'
    st.markdown(html, unsafe_allow_html = True)

else:
    st.header('運用協同過濾技術於表演藝術購票推薦之研究 ─以兩廳院售票資料為例    前端展示')
    
    st.markdown('<a href = "https://etds.lib.tku.edu.tw/ETDS/Home/Detail/U0002-2106202315095100" target = "_blank">論文全文下載</a>', unsafe_allow_html = True)
       
    st.divider()
    
    st.write('於實驗中建立兩個推薦系統：')
    
    ls_1 = ['傳統的以物品為基礎的**協同過濾模型** (Item-item Collaborative Filtering model)，以 cosine similariy 計算音樂會間的相似度。',
            '利用深度學習的**神經協同過濾模型** (Neural Collaborative Filtering, NCF)。']
    
    v_1 = ''
    n_1 = 1
    
    for l in ls_1:
        v_1 += str(n_1) + '. ' + l + '\n'
        n_1 += 1
    
    st.markdown(v_1)
    
    
    
    st.write('實驗用資料集為兩廳院售票系統2011-2019年間，系統會員購買國家兩廳院演奏廳節目票券的交易紀錄。（非公開）')
    
    st.write('實驗結果為建立推薦表現比傳統協同過濾模型要好的 NCF 模型，並找出以下影響模型表現的特徵：')
    
    ls_2 = ['下訂日與音樂會相距日數',
          '音樂會標題',
          '使用者最近一次下訂時間與訓練日期相距日數',
          '音樂會類別',
          '音樂會為國內或國外節目',
          '音樂會發生於星期幾']
    
    v_2 = ''
    
    for l in ls_2:
        v_2 += '- ' + l + '\n'
    
    st.markdown(v_2)
    
    st.divider()
    
    st.header('說明：')
    
    st.subheader('訓練時間')
    st.write('實驗將資料集依照「訓練時間」切分為訓練集與測試集。訓練時間共有 23個，第一個訓練時間為 2019/01/01。之後每 15日設定一個訓練時間，例如，第二個訓練時間為 2019/01/16，第三個為 2019/01/31，依此類推。')
    st.write('早於訓練時間的資料為訓練集，訓練時間後的資料為測試資料。測試集自測試資料中挑出。共有 23個訓練時間，故有 23組訓練集與測試集。')
    
    st.subheader('模型編號')
    st.write('使用第一個訓練時間對應的訓練集所建立的模型，該模型編號為 01 (model 01)，第二個訓練時間對應的模型編號為 02 (model 02)，依此類推。')
    
    st.subheader('使用者代碼')
    st.write('自左側選單選擇模型編號後，可輸入使用者代碼，兩個模型分別產生對該使用者提出長度為 10 的推薦清單。使用者代碼為經過轉換的流水編號，非該使用者的「會員代碼」。意即，model 02中，使用者代碼為 1者，並不一定與 model 12中，使用者代碼為 1者為同一人。')
    st.write('※可利用畫面左上角的 > 和 X 符號開啟 / 收納左側選單')
    
    st.divider()
    
    st.subheader('幾個特殊推薦結果')
    st.write('1. **NCF 表現好、傳統 CF 表現不好**：model 12，使用者代碼 10')
    st.write('使用者於訓練集中僅有一筆交易資料，NCF 仍能正確將使用者確實購買的節目納入清單中。傳統 CF提出的推薦則是相當分散。')
    
    st.write('2. **傳統 CF 表現好、NCF 表現不好**：model 15，使用者代碼 237')
    st.write('觀察使用者過去購買紀錄，可發現該使用者傾向觀賞聲樂相關節目。傳統 CF 模型正確捕捉此特性，並成功將使用實際購買節目納入推薦清單中。NCF 的推薦則以 NSO 節目為主，模型沒有學到「聲樂」為該使用者的興趣。')
    
    st.write('3. **兩個模型表現都很好**：model 02，使用者代碼 44')
    st.write('使用者於訓練集中的購買紀錄，多為熱門節目；於測試集中的購買紀錄，也多是熱門節目。節目交易筆數多時，模型能做出較準確的推薦。')
    
    st.write('4. **兩個模型表現都不好**：model 16，使用者代碼 60')
    st.write('此為使用者過去觀賞過的節目包羅萬象，即便是有經驗的行銷人員也很難自觀賞紀錄看出使用者的喜好。觀察使用者測試集中實際購買的節目，也是各類別都有，對兩個模型來說，要能成功為該使用者提出準確的推薦，相當困難。')        
