<h2>運用協同過濾技術於表演藝術購票推薦之研究 ─以兩廳院售票資料為例</h2>
<p>碩論，<a href = "https://etds.lib.tku.edu.tw/ETDS/Home/Detail/U0002-2106202315095100", target = "_blank">全文下載</a>。</p>
<p>建立兩個推薦系統：</p>
<ul>
  <li>傳統的以物品為基礎的協同過濾模型 (Item-item Collaborative Filtering model)，以 cosine similariy 計算音樂會間的相似度。</li>
  <li>利用深度學習的<a href = "https://doi.org/10.1145/3038912.3052569" target = "_blank">神經協同過濾模型 (Neural Collaborative Filtering, NCF)。</a></li>
</ul>
<p>實驗用資料集為兩廳院售票系統2011-2019年間，系統會員購買國家兩廳院演奏廳節目票券的交易紀錄。（非公開）</p>
<p>實驗結果為建立推薦表現比傳統協同過濾模型要好的 NCF 模型，並找出以下影響模型表現的特徵：</p>
<ul>
  <li>下訂日與音樂會相距日數</li>
  <li>音樂會標題</li>
  <li>使用者最近一次下訂時間與訓練日期相距日數</li>
  <li>音樂會類別</li>
  <li>音樂會為國內或國外節目</li>
  <li>音樂會發生於星期幾</li>
</ul>
<p>Streamlit 前端展示：<a href = "https://ntch-recommender.streamlit.app/" target = "_blank">https://ntch-recommender.streamlit.app/</a></p>
