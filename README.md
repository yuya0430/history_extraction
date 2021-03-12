# 重要履歴抽出

### <ファイル名，フォルダ名の意味>

2history ... 前の発話を2つ加えた直近3発話入力

3history ... 前の発話を3つ加えた直近4発話入力

full ... 評価指標の変化を捉えるために学習途中のモデルを多く保存したやつ(間に合わなかったので論文には未使用)

◯◯◯&△△△ ... 複数の対話行為タグを使用したモデル

24 ... 実験②の従来の履歴使用法との比較を行った際の結果

train ．．． 学習用

predict ... 予測用

evaluation ... 評価用

-------------------------------------------------------------------
### <フォルダ説明>

data ... SGDデータセット

multi_cased_L-12_H-768_A-12 ... 事前学習済みBERTモデル

data_check_script ... データや結果の分析に用いたスクリプト

output_??? ... 評価実験の出力

schema_guided_dst_??? ... 対話状態追跡モデルのスクリプト

???.sh ... abci用のシェルスクリプト

----------------------------------------------------------------
### <実行方法>

(実行条件)

python  /3.6/3.6.5

cuda    /10.0/10.0.130.1

cudnn   /7.4/7.4.2

tensorflow	1.14.0


(dataのダウンロード)
```
https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
```

(BERT のダウンロード)
```
https://github.com/google-research/bert
```

BERT-Base, Cased か BERT-Base, Multilingual Cased (New, recommend) をクリック

今回の実験では後者を使用

(abci で実行する場合)
```
"qsub -g gcb50327 ???.sh"
```

gcb50327 はグループのID

(abci 以外で実行する場合)

1． pip -r requirements.txt

2.
(学習時)
```
python -m schema_guided_dst.baseline.train_and_predict --bert_ckpt_dir <multi_cased_L-12_H-768_A-12のアドレス> --dstc8_data_dir <dataのアドレス> --dialogues_example_dir <対話データの保存先> --schema_embedding_dir <スキーマ埋め込みの保存先> --output_dir <出力結果の保存先> --dataset_split train --run_mode train --task_name dstc8_single_domain > sgd.log 2>&1
```

(予測時)
```
python -m schema_guided_dst.baseline.train_and_predict --bert_ckpt_dir <multi_cased_L-12_H-768_A-12のアドレス> --dstc8_data_dir <dataのアドレス> --dialogues_example_dir <対話データの保存先> --schema_embedding_dir <スキーマ埋め込みの保存先> --output_dir <出力結果の保存先> --dataset_split dev --run_mode predict --task_name dstc8_single_domain --eval_ckpt 0,100000,206470
```

--eval_ckpt には出力したモデルのチェックポイントの番号を入れる

出力結果を保存するフォルダに予測結果を持つ "pred_res_?" フォルダが追加される 

(評価時)
```
python -m schema_guided_dst.evaluate --dstc8_data_dir <dataのアドレス> --prediction_dir <pred_res_?のアドレス> --eval_set dev --output_metric_file <評価結果のファイル名>
```

これと実際のシェルスクリプトを比較すればコマンドの意味が分かると思います．

### パラメータについて

max_seq_length : 2発話使用のとき 80, 3発話使用のとき 120, 4発話使用の時, 160

train_batch : 実験①の対話行為ごとの比較を行うときは32，実験②の従来の履歴使用法との比較を行うときは24とした．
