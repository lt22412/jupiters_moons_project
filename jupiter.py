import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
import numpy as np


class Moons:

    def __init__(self, path_to_db):

        connect= sqlite3.connect(path_to_db)
        query='SELECT * FROM moons'

        df_0 =pd.read_sql_query(query, connect)
        connect.close()
        
        self.data = df_0
        self.data=self.data.set_index('moon')

        self.create_group_mapping()# moving groups to numbers (treating groups as numbers is more effective further)
        
        self.data=self.data.drop(['group'], axis=1)# drop the word group name column as we have the group_to_number dictionary attached if we need to recall the group name from its number.

        


    def create_group_mapping(self):

        """ Create and apply a mapping from group names to numbers. """
        
        if 'group_number' in self.data.columns.tolist():
            print("Group number is already a feature in the dataset")
            return

        unique_groups = self.data['group'].unique()
        self.group_to_number = {group: i for i, group in enumerate(unique_groups)}

        # Replace group names with numbers in the dataset
        self.data['group_number'] = self.data['group'].map(self.group_to_number)

        return



    def summary_statistics(self):
        """ Return summary statistics of the dataset. """
        return self.data.describe()


    def get_features(self):
        """ Return the names of all columns in the dataset. """
        return self.data.columns.tolist()
    
    def get_moons(self):
        """ Return a list of moon names from the dataset. """
        # Replace 'moon_name_column' with the actual column name that contains moon names
        return self.data.index.tolist()
    
    def get_moon_count(self):
        """ Return the number of moons in the dataset. """
        return len(self.data)
    
    def get_moon_data(self, moon_name):
        """ Return data for a specific moon. """
        return self.data.loc[moon_name]


   

    def count_missing_values(self, axis='col'):
        """
        Count missing values in the dataset.
        Args:
        axis (str): 'col' for column-wise or 'row' for row-wise missing value count.

        Returns:
        dict: Dictionary with column/row names as keys and count of missing values as values.
        """
        if axis not in ['col', 'row']:
            raise ValueError("Axis must be 'col' or 'row'")

        missing_values_dict = {}
        if axis == 'col':
            # Count missing values in each column
            missing_values_dict = self.data.isnull().sum().to_dict()
        elif axis == 'row':
            # Count missing values in each row
            for row in self.data.index:
                missing_values_dict[row] = self.data.loc[row].isnull().sum()

        return missing_values_dict
    
    
    
    def calculate_correlation_matrix(self, axis='col'):
        """
        Calculate the correlation matrix for the specified columns.

        Args:
        columns (list, optional): A list of column names to include in the correlation matrix.
                                   If None, all numerical columns will be used.

        Returns:
        pd.DataFrame: A DataFrame representing the correlation matrix.
        """
        
        
        # Select only numerical columns for correlation matrix
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        needed_data= self.data[numeric_cols]
        
        if axis== 'row':
            needed_data=needed_data.T

        correlation_matrix = needed_data.corr()


        return correlation_matrix
    
    def visualise_corr(self,axis='col'):
        
        corr_matrix=self.calculate_correlation_matrix(axis)
        
        annot=True
        statement='Features correlation Matrix'
        if(axis=='row'):
            annot=False
            statement='Moons correlation Matrix'

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool),k=1)
        plt.figure(figsize=(20, 20))

        
        sns.heatmap(corr_matrix, 
            annot=annot,
            fmt='.1f',
            cmap='coolwarm',
            mask=mask,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .75, "label": "Correlation Value"},
            ) 



        

        plt.title(statement)

        # Display the plot
        plt.show()

        return

    def plot_distribution(self, column):
        """ Plot distribution of a specified column. """
        self.data[column].hist()
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def feature_histograms(self):
        n_rows = 2
        n_cols = 4


        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))  # Adjust the figure size as needed
        

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # List of your dataset's feature names
        feature_names = self.data.columns.tolist()
        # Loop through the features and create a histogram for each
        for i, feature in enumerate(feature_names):
            # Select the axis where the histogram will be plotted
            ax = axes[i]
    
            # Plot the histogram
            
            if feature== "group":
                continue  #this is due to it being broken and unnecessary. group_number is the same but sorted and more informative.

            if feature == "group_number":

                ax.hist(self.data[feature],bins=7, color='blue', edgecolor='black')
                ax.set_title(f'{feature}')

                continue
            
            ax.hist(self.data[feature], color='blue', edgecolor='black')
            # Set the title with the feature name
            ax.set_title(f'{feature}')

        # Adjust the layout so titles and labels don't overlap
        plt.tight_layout()
        # Show the plot
        plt.show()
        

    def feature_box_plots(self):
        n_rows = 2
        n_cols = 4


        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))  # Adjust the figure size as needed
        

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        
        
        # List of your dataset's feature names
        feature_names = self.data.columns.tolist()
        # Loop through the features and create a histogram for each

        for i, feature in enumerate(feature_names):
            
            if feature== "group":
                continue  #this is due to it being broken and unnecessary. group_number is the same but sorted and more informative.

            ax = axes[i]
    
            # Draw the box plot using seaborn
            sns.boxplot(self.data[feature], ax=ax)

            # Set the title with the feature name
            ax.set_title(f'{feature}')

        # Adjust the layout so titles and labels don't overlap
        plt.tight_layout()
        # Show the plot
        plt.show()


