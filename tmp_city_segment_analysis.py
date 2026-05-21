import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

file_path = 'c:/Users/ADMIN/OneDrive/Documents/AI-ACCELERATOR_PROGRAM/week_05/day_01/03_lab/konga_transactions.csv'

df = pd.read_csv(file_path)
df['order_date'] = pd.to_datetime(df['order_date'])
ref_date = df['order_date'].max() + pd.Timedelta(days=1)
cm = df.groupby('customer_id').agg({'order_date': lambda x: (ref_date - x.max()).days, 'order_id':'count', 'line_total_ngn':'sum'})
cm['aov'] = cm['line_total_ngn'] / cm['order_id']
cm['orders_per_month'] = cm['order_id'] / 12
cm.columns = ['recency','frequency','monetary','aov','orders_per_month']

X_scaled = StandardScaler().fit_transform(cm)
cm['cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)

cluster_means = cm.groupby('cluster')['monetary'].mean().sort_values(ascending=False)
print('cluster_means:\n', cluster_means)

top_cluster = cluster_means.index[0]
latest_city = df.sort_values('order_date').groupby('customer_id').last()['city']

top_customers = cm[cm['cluster']==top_cluster].index
city_counts = latest_city.loc[top_customers].value_counts()
city_prop = city_counts / city_counts.sum()
print('\nTop cities in highest-value segment:')
print(pd.concat([city_counts.head(10), city_prop.head(10)], axis=1, keys=['count','pct']))

overall_prop = df.groupby('customer_id').last()['city'].value_counts(normalize=True)
print('\nOverall city proportions (top 10):')
print(overall_prop.head(10))
