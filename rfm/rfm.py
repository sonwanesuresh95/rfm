import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class RFM:
    """
    Performs RFM analysis and customer segmentation on input dataset.
    
    Attributes:
    -----------
    customer_df : dataframe with unique customers with their rfm values/scores
    segment_df : dataframe with count of customers across all segments

    Parameters:
    -----------
    customer_id : string, name of the column by which individual customer is identified
    transaction_date : string, name of the column which represents trasaction date
    amount : string, column stating amount of transaction
    automated : bool, default=True, carries out operations automatically; pass False if you want to perform each operation manually
    """
    def __init__(self, df:pd.DataFrame, customer_id:str, transaction_date:str, amount:str, automated=True):
        self.df = df
        self.customer_id = customer_id
        self.transaction_date = transaction_date
        self.amount = amount
        
        # automated operations
        if automated:
            df_grp = self.produce_rfm_dateset(self.df)
            df_grp = self.calculate_rfm_score(df_grp)
            self.customer_df = self.find_segments(df_grp)
            self.segment_df = self.find_segment_df(self.customer_df)
        
    def produce_rfm_dateset(self, df:pd.DataFrame)->pd.DataFrame:
        """
        produce_rfm_dataset(df)
        |  
        |  Finds RFM values for entered dataset and returns a dataframe object.
        |  functionality consists of preprocessing, grouping by customer_id, finding RFM values.
        |  
        |  Parameters:
        |  -----------
        |  df : pd.DataFrame object, containing raw transaction records of customers
        |  
        |   Returns
        |   -------
        |       DataFrame object
        """
        for col in df.columns:
            if col != self.customer_id:
                df[col] = df[col].astype(str).apply(lambda x: x.strip()[:-2] if '.0' in x else x.strip())
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col])
        
        df = df.sort_values(by=self.transaction_date)
        df[self.amount] = df[self.amount].apply(lambda x: float(x) if x not in ['','nan'] else 0)
        df = df.dropna(subset=[self.customer_id,self.amount])
        df = df.drop_duplicates()
        df = df.reset_index().drop(columns=['index'], axis=1)
        df[self.customer_id] = df[self.customer_id].astype(str).apply(lambda x: x.strip()[:-2] if '.0' in x else x.strip())

        # grouping by customer_id
        df_grp = df[[self.customer_id,self.transaction_date,self.amount]].groupby(self.customer_id,).agg(list).reset_index()
        
        
        # finding r,f,m values
        
        latest_date = df[self.transaction_date].max()
        df_grp['recency'] = df_grp[self.transaction_date].apply(lambda x: (latest_date - x[-1]).days)
        df_grp['frequency'] = df_grp[self.amount].apply(len)
        df_grp['monetary_value'] = df_grp[self.amount].apply(sum)
        
        return df_grp[[self.customer_id, 'recency', 'frequency', 'monetary_value']]
    
    def calculate_rfm_score(self, df:pd.DataFrame)->pd.DataFrame:
        """
        calculate_rfm_score(df)
        |
        |  calculates rfm scores based on rfm values (binning)
        |
        |  Parameters:
        |  -----------
        |  df : pd.DataFrame object, containing recency, frequency and monetary_value columns
        |
        |  Returns
        |  -------
        |  df : pd.DataFrame object
        """
        df['r'] = pd.qcut(df['recency'].rank(method='first'),5,labels=[5,4,3,2,1]).tolist()
        df['f'] = pd.qcut(df['frequency'].rank(method='first'),5,labels=[1,2,3,4,5]).tolist()
        df['m'] = pd.qcut(df['monetary_value'].rank(method='first'),5,labels=[1,2,3,4,5]).tolist()
        df['rfm_score'] = df['r'].apply(str) + df['f'].apply(str) + df['m'].apply(str)
        df = df.sort_values(by='rfm_score',ascending=False).reset_index(drop=True)
        return df
        
    def find_segments(self, df:pd.DataFrame)->pd.DataFrame:
        """
        find_segments(df)
        |
        |  finds customer segments based on the rfm scores
        |
        |  Parameters:
        |  -----------
        |  df : pd.DataFrame object, containing r,f,m scores columns
        |
        |  Returns
        |  -------
        |  df : pd.DataFrame object
        """
        classes = []
        for row in df.iterrows():
            if (row[1]['r'] in [4,5]) & (row[1]['f'] in [4,5]) & (row[1]['m'] in [4,5]):
                classes.append({row[1][self.customer_id]:'Champions'})
            elif (row[1]['r'] in [4,5]) & (row[1]['f'] in [1,2]) & (row[1]['m'] in [3,4,5]):
                classes.append({row[1][self.customer_id]:'Promising'})
            elif (row[1]['r'] in [3,4,5]) & (row[1]['f'] in [3,4,5]) & (row[1]['m'] in [3,4,5]):
                classes.append({row[1][self.customer_id]:'Loyal Accounts'})
            elif (row[1]['r'] in [3,4,5]) & (row[1]['f'] in [2,3]) & (row[1]['m'] in [2,3,4]):
                classes.append({row[1][self.customer_id]:'Potential Loyalist'})
            elif (row[1]['r'] in [3,4,5]) & (row[1]['f'] in [2,3,4,5]) & (row[1]['m'] in [1,2]):
                classes.append({row[1][self.customer_id]:'Low Spenders'})
            elif (row[1]['r'] in [5]) & (row[1]['f'] in [1]) & (row[1]['m'] in [1,2,3,4,5]):
                classes.append({row[1][self.customer_id]:'New Active Accounts'})
            elif (row[1]['r'] in [2,3]) & (row[1]['f'] in [1,2]) & (row[1]['m'] in [4,5]):
                classes.append({row[1][self.customer_id]:'Need Attention'})
            elif (row[1]['r'] in [2,3]) & (row[1]['f'] in [1,2]) & (row[1]['m'] in [1,2,3]):
                classes.append({row[1][self.customer_id]:"About to Sleep"})
            elif (row[1]['r'] in [1,2]) & (row[1]['f'] in [1,2,3,4,5]) & (row[1]['m'] in [3,4,5]):
                classes.append({row[1][self.customer_id]:'At Risk'})
            elif (row[1]['r'] in [1,2]) & (row[1]['f'] in [1,2,3,4,5]) & (row[1]['m'] in [1,2]):
                classes.append({row[1][self.customer_id]:"Lost"})
            else:
                classes.append({0:[row[1]['r'],row[1]['f'],row[1]['m']]})
        accs = [list(i.keys())[0] for i in classes]
        segments = [list(i.values())[0] for i in classes]
        df['segment'] = df[self.customer_id].map(dict(zip(accs,segments)))
        return df
    
    def find_segment_df(self, df:pd.DataFrame)->pd.DataFrame:
        """
        find_segment_df(df)
        |
        |  returns segment distribution dataset
        |
        |  Parameters:
        |  -----------
        |  df : pd.DataFrame, customer_df, result from find_segments function
        |
        |  Returns segment distribution dataframe
        """
        segment_df = df[['segment',self.customer_id]].groupby('segment',sort=False).count().reset_index().rename({self.customer_id:'no of customers'},axis=1)
        return segment_df
    
    def find_customers(self, segment:str)->pd.DataFrame:
        """
        find_customers(segment)
        |
        |  returns dataframe of entered segment
        |
        |  Parameters:
        |  ----------
        |  segment : str, one of the 10 categories : ['Champions', 'Loyal Accounts', 'Low Spenders', 'Potential Loyalist', 'Promising', 'New Active Accounts', 'Need Attention', 'About to Sleep', 'At Risk', 'Lost']
        |
        |  Returns dataframe of customers with specified segment
        """
        return self.customer_df[self.customer_df['segment'] == segment].reset_index(drop=True)
    
    def distribution_plot(self, column:str, figsize=(10,5)):
        """
        distribution_plot(col, figsize)
        |
        |  shows distribution of entered column
        |
        |  Parameters:
        |  -----------
        |  column : str, column name
        |  figsize : (x,y) size of the figure; default = (10,5)
        |
        |  Returns None
        """
        plt.figure(figsize=figsize)
        plt.hist(self.customer_df[column], edgecolor='black')
        plt.grid()
        plt.xlabel(column.capitalize())
        plt.ylabel('Count')
        plt.title(column.capitalize())
        plt.show()
    
    def versace_plot(self, column1:str, column2:str, figsize=(10,7)):
        """
        versace_plot(column1, column2, figsize)
        |
        |  shows scatterplot of 2 columns : col1 vs col2
        |
        |  Parameters:
        |  -----------
        |  column1 : str, column 1 name
        |  column2 : str, column 2 name
        |
        |  figsize : (x,y) size of the figure; default = (10,7)
        |
        |  Returns None
        """
        plt.figure(figsize=figsize)
        plt.scatter(self.customer_df[column1],self.customer_df[column2])
        plt.xlabel(column1.capitalize())
        plt.ylabel(column2.capitalize())
        plt.title(f'{column1.capitalize()} vs {column2.capitalize()}')
        plt.grid()
        plt.show()
        
    def segment_distribution(self):
        """
        segment_distribution()
        |
        |  shows no. of customers across all segments
        |
        |  Parameters: None
        |
        |  Returns None
        """
        x = self.segment_df['segment'][::-1]
        y = self.segment_df['no of customers'][::-1]
        plt.figure(figsize=(8,5))
        plt.barh(x,y,color=['indigo','blueviolet','royalblue','steelblue','darkblue','lime','forestgreen','yellow','orange','red'],edgecolor='black')
        plt.xlabel('No of Customers')
        plt.ylabel('Segment')
        plt.title('Segment Distribution')
        plt.grid()
        plt.show()
        
    def distribution_by_segment(self, column:str, take='median'):
        """
        distribution_by_segment(column, take)
        |
        |  shows mean/median distribution of column by segment
        |
        |  Parameters:
        |  -----------
        |  column : str, column name
        |  take : str, mean or median; include whichever required
        |
        |  Returns None
        """
        med = self.customer_df.groupby('segment',sort=False)[column].agg(lambda x: list(x)).reset_index()
        if take == 'median':
            med[column] = med[column].apply(np.median)
        if take == 'mean':
            med[column] = med[column].apply(np.mean)
        x = med['segment'][::-1]
        y = med[column][::-1]
        plt.figure(figsize=(8,5))
        plt.barh(x,y,color=['indigo','blueviolet','royalblue','steelblue','darkblue','lime','forestgreen','yellow','orange','red'],edgecolor='black')
        plt.xlabel(f'{take} {column.capitalize()}')
        plt.ylabel('Segment')
        plt.title(f'{take} {column.capitalize()} by Segment')
        plt.grid()
        plt.show()
        
    def rfm_histograms(self):
        """
        rfm_histograms()
        |
        |  shows histograms of r,f,m
        |
        |  Parameters: None
        |
        |  Returns None
        """
        fig, axs = plt.subplots(1,3)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        fig.suptitle('RFM Distributions')
        axs[0].hist(self.customer_df['recency'],edgecolor='black')
        axs[0].set_title('Recency')
        axs[1].hist(self.customer_df['frequency'],edgecolor='black')
        axs[1].set_title('Frequency')
        axs[2].hist(self.customer_df['monetary_value'],edgecolor='black')
        axs[2].set_title('Monetary Value')
        plt.show()