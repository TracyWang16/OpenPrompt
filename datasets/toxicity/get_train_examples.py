import pandas as pd
import os

toxicity_file_name = 'datasets/toxicity/data_toxicity_lg05.csv'
save_toxicity_file_name = 'datasets/toxicity/train_toxicity_lg05.txt'

nontoxicity_file_name = 'datasets/toxicity/data_nontoxicity_eq0.csv'
save_nontoxicity_file_name = 'datasets/toxicity/train_nontoxicity_eq0.txt'

df_toxicity = pd.read_csv(toxicity_file_name)
df_nontoxicity = pd.read_csv(nontoxicity_file_name)
#df = df[0:2]

df_toxicity_train = list( df_toxicity.groupby('split'))[1][1]
df_nontoxicity_train =list( df_nontoxicity.groupby('split'))[1][1]

toxicity_len = len(df_toxicity_train)
nontoxicity_len = len(df_nontoxicity_train)

dataset_len = min(toxicity_len,nontoxicity_len)

df_nontoxicity_train = df_nontoxicity_train.sample(n=dataset_len)
df_toxicity_train = df_toxicity_train.sample(n=dataset_len)


#assert not os.path.exists(save_toxicity_file_name)
with open(save_toxicity_file_name,'w') as f_write:
	for text in df_toxicity_train['comment_text']:
		f_write.write(text.replace('\n','')+'\n')
	f_write.close()
i = 0
#assert not os.path.exists(save_nontoxicity_file_name)
with open(save_nontoxicity_file_name,'w') as f_write:
	for text in df_nontoxicity_train['comment_text']:
		i = i+1
		f_write.write(text.replace('\n','')+'\n')
	f_write.close()

aa = 0


