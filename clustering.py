#クラスタリング結果を作成して出力する処理の説明


import pandas as pd
import subprocess
import os

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np

#=============================================
# Inputファイル情報
#=============================================
INPUT_folder = "2_data"        
INPUT_DNAME = "サンプルデータ.csv"
#=============================================
# Outputファイル情報
#=============================================
OUTPUT_folder = "3_output"
#=============================================
# カレントパス
#=============================================
current_dpath = os.getcwd()
print("INFO:カレントパス:" + current_dpath)

#=============================================
# パレントパス
#=============================================
parent_dpath =os.path.sep.join(current_dpath.split(os.path.sep)[:-1])

print("INFO:パレントパス:" + parent_dpath)   

#=============================================
# Inputデータファイル Path
#=============================================
input_dpath =os.path.sep.join([parent_dpath + '\\' + INPUT_folder,INPUT_DNAME])
print("INFO:データファイルパス:" + input_dpath) 

#=============================================
# Outputデータファイル Path
#=============================================
output_dpath =parent_dpath + '\\' + OUTPUT_folder
print("INFO:出力先のフォルダパス:" + output_dpath)   

#=============================================
# サンプルデータ読み込む
#=============================================
df = pd.read_csv(input_dpath,encoding='shift-JIS')
df = pd.DataFrame(df)

#=============================================
# 分割の閾値設定
#=============================================
# ① y軸のクラスタ分割位置を指定　
threshold_distance = 1500

# ② クラスタ数指定　 ：　8ケ
criter = 8

#=============================================
# linkageの計算
#============================================
Z = linkage(df.iloc[:, 1:4], method="ward")

#=============================================
# 最大距離　適切な閾値の目安が得られます。
#=============================================
#print("最大距離:", np.max(Z[:, 2]))

#=============================================
# デンドログラムを描画
#=============================================
dendro = dendrogram(Z, labels=df["顧客ID"].values)

#=============================================
# 閾値に赤い破線のラインを入れる
#=============================================
plt.axhline(y=threshold_distance, color='red', linestyle='--')

#=============================================
# クラスタリング画　保存
#=============================================
plt.rcParams["font.size"] = 10
plt.savefig(output_dpath + "\\クラスタリング.png")
plt.show()

# 顧客ID　デンドログラムと同じ順番
#=============================================
#  クラスタのラベルをつける
#=============================================
labels_distance = fcluster(Z, t=threshold_distance, criterion="distance")

#=============================================
# 元のデータにクラスタ番号をくっつける
#=============================================
df_distance = df.copy()
# cluster列を追加し、それぞれの行が属するクラスタ番号を書き込む
df_distance["cluster"] = labels_distance

#=============================================
# デンドログラムで並んでいた順に並べ替える 
#=============================================
ordered_df_distance = df_distance.iloc[dendro["leaves"]].reset_index(drop=True)

#=============================================
# CSV出力
#=============================================-
ordered_df_distance.to_csv(
    output_dpath + '\\クラスタリング結果_距離' + str(threshold_distance) + '_並べ替え.csv',
    index=False,
    encoding="utf-8-sig"
)
ordered_df_distance.head(3)


# 顧客ID　元のデータと同じ順番
#=============================================-
#  クラスタのラベルをつける
#=============================================-
labels_maxclust = fcluster(Z, t=criter, criterion="maxclust")


#=============================================
# 元のデータにクラスタ番号をくっつける
#=============================================
df_maxclust = df.copy()
# cluster列にクラスタ番号追加
df_maxclust["cluster"] = labels_maxclust

#=============================================-
# CSV出力
#=============================================-
df_maxclust.to_csv(
    output_dpath + '\\クラスタリング結果_クラスタ数_' + str(criter) + '_.csv',
    index=False,
    encoding="utf-8-sig"
)
df_maxclust.head(3)

