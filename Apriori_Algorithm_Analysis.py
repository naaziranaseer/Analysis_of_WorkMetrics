import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt

company_file = "companies.csv"
data = pd.read_csv(company_file, encoding='latin1')
print(data.head())
print(data.columns)
print(data.info())
print(data.company_name.unique())

def convert_k_to_num(val):
    if isinstance(val, str):
        val = val.lower().replace('k', 'e3').replace(',','')
    try:
        return float(val)
    except ValueError:
        return val
    
data['reviews'] = data['reviews'].apply(convert_k_to_num)
data['jobs'] = data['jobs'].apply(convert_k_to_num)
data['interviews'] = data['interviews'].apply(convert_k_to_num)
print(data[['reviews','jobs','interviews']].head())

data['highly_rated_for'] = data['highly_rated_for'].fillna('')
data['critically_rated_for'] = data['critically_rated_for'].fillna('')

data['highly_rated_for'] = data['highly_rated_for'].str.split(',')
data['critically_rated_for'] = data['critically_rated_for'].str.split(',')

data['items'] = data['highly_rated_for']+data['critically_rated_for']

items_list = list(set([item for sublist in data['items'] for item in sublist if item]))

basket = pd.DataFrame(0, index = data['company_name'], columns= items_list)

for idx, row in data.iterrows():
    for item in row['items']:
        if item:
            basket.loc[row['company_name'],item] = 1

min_support = 0.1
frequent_itemsets = apriori(basket, min_support = min_support, use_colnames = True)
print(frequent_itemsets)

min_confidence = 0.4
rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = min_confidence, num_itemsets= 2)
print (basket.sum(axis=0).sort_values(ascending = False))
print("Items List:", items_list)
print(data['items'])
print("Sample of 'items' column:")
print(data['items'].head())

if not rules.empty:
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    apriori_table = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']]

    print("Final Apriori Table:")
    print(apriori_table)
  
    apriori_table.to_csv("apriori_table.csv", index=False)
else:
    print("No association rules were generated. Consider lowering the support or confidence thresholds.")

import matplotlib.pyplot as plt

if not apriori_table.empty:
    apriori_table['rule'] = apriori_table['antecedents'] + " → " + apriori_table['consequents']
    
    top_rules = apriori_table.sort_values(by='support', ascending=False).head(10)

    plt.figure(figsize=(8, 8),facecolor='white')
    wedges, texts, autotexts = plt.pie(
        top_rules['support'],
        labels=top_rules['rule'],
        autopct='%1.1f%%',  
        startangle=140,
        colors=sns.color_palette('pastel'),
        textprops={'color': 'black', 'fontsize': 12, 'fontweight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('black')  
        autotext.set_fontsize(14)     
        autotext.set_fontweight('bold') 

    plt.title('Analysis on Work Attributes', fontsize=35, color = 'black', fontweight='bold', fontfamily='Arial')

    plt.show()
else:
    print("No rules available for visualization.")