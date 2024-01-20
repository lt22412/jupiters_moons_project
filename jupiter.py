import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class Moons: #attributes: data (input dataset), feature_dictionary(off by default. It appears when we use method numerical_categorical_mapping , see function below).

    def __init__(self, path_to_db):# class constructor, path_to_db: string

        connect= sqlite3.connect(path_to_db)#connect to database with provided path
        query='SELECT * FROM moons'#get all tables from database

        df_0 =pd.read_sql_query(query, connect)#get the dataset in
        connect.close()#disconnect from database
        
        self.data = df_0
        self.data=self.data.set_index('moon')
        
    def numerical_categorical_mapping(self, feature):#Create and apply a mapping for a categorical literal feature to numbers.

        if feature not in self.data.columns:# check if feature is present
            print("This feature is not found in the dataset")
            return
        
        unique_examples=self.data[feature].unique()# set of unique objects within feature
        
        dict = {group: i for i,group in enumerate(unique_examples)}# create a dictionary in form  {unique objects: their id's in new feature}
        
        dictionary_attribute_name=feature+'_dictionary'#set name of the attribute

        setattr(self,dictionary_attribute_name,dict)# add an attribute with name feature_number('feature' is the name of the feature) that holds dictionary so we know what numbers in new numerical feature mean.

        self.data[feature]=self.data[feature].map(getattr(self,dictionary_attribute_name))# Replace feature names with numbers in the dataset
        
        return

    def summary_statistics(self):#Return summary statistics of the dataset.
        return self.data.describe()
    
    def get_features(self):#Return the names of all columns in the dataset.
        return self.data.columns.tolist()
    
    def get_moons(self):# Return a list of moon names from the dataset. 
        return self.data.index.tolist() 
    
    def get_moon_count(self):#Return the number of moons in the dataset. 
        
        return len(self.data)
    
    def get_moon_data(self, moon_name):#Return data for a specific moon. moon_name: string

        if moon_name in self.data.index:# check if moon is in the dataset
            return self.data.loc[moon_name]#return corresponding row
        else:
            print('moon not found')# else return error
            return None

    def get_feature_types(self):# return dictionary with data types of every feature
        return self.data.dtypes.apply(lambda x: x.name).to_dict()

    def count_missing_values(self, axis='col'):#Count missing values in the dataset. Returns a dictionary with missing values for each feature. axis: string
        if axis not in ['col', 'row']:#if axis name wrong return error
            raise ValueError("Axis must be 'col' or 'row'")

        missing_values_dict = {}
        if axis == 'col':
            missing_values_dict = self.data.isnull().sum().to_dict()# Count missing values in each column
        elif axis == 'row':
            for row in self.data.index:
                missing_values_dict[row] = self.data.loc[row].isnull().sum()# Count missing values in each row

        return missing_values_dict
    
    def calculate_correlation_matrix(self, axis='col'):#Calculate the correlation matrix for the specified columns. axis: string. Returns a dataframe representing the correlation matrix.
        
        numeric_cols = self.data.select_dtypes(include=['number']).columns# Select only numerical columns for correlation matrix
        needed_data= self.data[numeric_cols]
        
        if axis== 'row':
            needed_data=needed_data.T# if want corr matrix for rows transpose 

        correlation_matrix = needed_data.corr()#get the corr matrix for columns


        return correlation_matrix
    
    def visualise_corr(self,axis='col',size=13):#Visualise the correlation matrix for the specified columns. axis: string, size: int
        
        corr_matrix=self.calculate_correlation_matrix(axis)#use previous function to calculate correlation matrix
        
        annot=True
        statement='features correlation matrix'
        if(axis=='row'):
            annot=False# for moons correlation values dont fit inside the squares, so set values to invisible so dont see them.
            statement='moons correlation matrix'

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool),k=1)#so corr matrix is lower triangle

        plt.figure(figsize=(size, size))#empty plot

        sns.heatmap(corr_matrix, #plot the matrix
            annot=annot,
            fmt='.1f',
            cmap='coolwarm',
            mask=mask,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .75, "label": "correlation Value"},
            ) 

        plt.title(statement)#set title to Moons correlation Matrix

        # Display the plot
        plt.show()

        return

    def plot_distribution(self, column):#Plot distribution of a specified column. Column: string
        
        if column not in self.data.columns:# check if column is in dataset
            print('No such column')
            return
        
        self.data[column].hist()# build histogram
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()# show histogram

    def feature_histograms(self):#Create a figure and a grid of subplots
        n_rows = 2# set dimensions of plot
        n_cols = 4

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10)) #adjust the figure size
        
        axes = axes.flatten()#flatten the axes array
       
        feature_names = self.data.columns.tolist()#List ofdataset's feature names
        
        for i, feature in enumerate(feature_names):#loop through the features and create a histogram for each
            
            ax = axes[i]# histogram will be plotted there

            if feature == "group":# Only 7 bins for group histogram as there are 7 histograms

                ax.hist(self.data[feature],bins=7, color='blue', edgecolor='black')# Plot the histogram
            
                ax.set_title(f'{feature}')

                continue
            
            ax.hist(self.data[feature], color='blue', edgecolor='black')# Plot the histogram
            
            ax.set_title(f'{feature}')# Set the title with the feature name

        
        plt.tight_layout()#adjust the layout so titles and labels don't overlap
        
        plt.show()#show the graph

    def feature_histograms_by_groups(self):#Create a figure and a grid of subplots by groups
        n_rows = 2# set dimensions of plot
        n_cols = 4


        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))  #adjust the figure size
        

        axes = axes.flatten()#flatten the axes array

        feature_names = self.data.columns.tolist()#list ofdataset's feature names
        
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown', 'pink']# colors we will use for groups

        df=self.data#for convenience switch datasets
        df['mass_kg'] = np.log1p(self.data['mass_kg'])#apply logarithmic transformation to large feature

        for i, feature in enumerate(feature_names):#loop through the features and create a histogram for each
            
            ax = axes[i]#axis for plotting
            
            if feature=='mass_kg':#just ignore because of the size
                continue

            for j in range(0,8):
                ax.hist(self.data[self.data.group==j][feature], color=colors[j], edgecolor='black', alpha=0.7, label=j)#Plot the histogram
            

            
            ax.set_title(f'{feature}')#set the title with the feature name
            

        
        plt.tight_layout()#adjust the layout
        
        plt.show()#show the plot
  
    def feature_box_plots(self):#Create a set of box plots for all features
        n_rows = 2# set dimensions of plot
        n_cols = 4

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))  #adjust the figure size
        
        axes = axes.flatten()#flatten the axes array

        
        feature_names = self.data.columns.tolist()#list ofdataset's feature names
        

        for i, feature in enumerate(feature_names):#loop through the features and create a histogram for each
            
            
            ax = axes[i]
    
            
            sns.boxplot(self.data[feature], ax=ax)#draw the box plot using seaborn

            
            ax.set_title(f'{feature}')#set the title with the feature name

       
        plt.tight_layout() #adjust the layout so titles look nice
        
        plt.show()#show the plot

    def plot_columns(self, column1, column2, use_group=False):# plot 2 columns against each other:column1,2: string, use_group: bool
        df=self.data#copy dataset for convenience
        if column1 in df.columns and column2 in df.columns and \
            pd.api.types.is_numeric_dtype(df[column1]) and \
            pd.api.types.is_numeric_dtype(df[column2]):# Check if both features are numerical

            plt.figure(figsize=(8, 6))#adjust the plot size
            

            if use_group and 'group' in df.columns:# Check if group is in the dataset
                groups=df.groupby('group')  # Group by group number
                for name, group in groups:
                    plt.scatter(group[column1], group[column2], alpha=0.5, label=name)# scatter every group with different color
                    plt.legend()
            else:
                plt.scatter(df[column1], df[column2], alpha=0.5)# Else scatter with same color


            plt.title(f'{column1} vs. {column2}')# Set titles and axis names(lower)
            plt.xlabel(column1)
            plt.ylabel(column2)
            plt.grid(True)
            plt.show()# Show the Plot
        else:
            print(f"Error: One or both columns are either not in the DataFrame or not numerical.")#return error if columns are wrong

    def linreg_subs(self, column1, column2):# Substitute missing values in column 2 with a linear regression predictions trained on column1 set of columns. Column1: list[strings], column2: string. 
        
        df=self.data#copy for convenience

        train_df = df[df[column2].notna()]#Split into train and predict based on NaN values
        predict_df = df[df[column2].isna()]

        
        model = LinearRegression()
        model.fit(train_df[column1], train_df[column2])# Train linear regression model

        
        predicted_values = model.predict(predict_df[column1])#predict the missing values in column1

       
        self.data.loc[self.data[column2].isna(), column2] = predicted_values#fill in the missing values in the original dataset

        train_predictions = model.predict(train_df[column1])
        r2 = r2_score(train_df[column2], train_predictions)# Estimate r2 and rmse 
        rmse = np.sqrt(mean_squared_error(train_df[column2], train_predictions))

        return [r2,rmse]# Return a list with r2 and rmse of model.
    

#a=Moons('jupiter.db')#little testing going on here
#a.numerical_categorical_mapping('group')
#print(a.data[a.data.group==1]['period_days'].head())
#print(a.linreg_subs(['radius_km'],'mag'))

#print(a.data [a.data['group']==0])
#print(a.group_dictionary)