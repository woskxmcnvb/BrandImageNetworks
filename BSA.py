import pandas as pd
import numpy as np
from itertools import product
import io

from ExcelReportBuilder import ExcelReportBuilder
from factor_analyzer import FactorAnalyzer

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore

import statsmodels.api as sm
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

def Standardize(df, across_all_dataframe=False): 
    if across_all_dataframe: 
        return df.sub(df.mean(axis=None)).div(df.stack().std())
    else:
        return df.sub(df.mean()).div(df.std())
    
def CalcLinearRegression(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame: 
    model = sm.OLS(y, X).fit()
    return pd.read_html(io.StringIO(model.summary().tables[1].as_html()), index_col=0, header=0)[0]

def VariablesSelection(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame: 
    model = LinearRegression()
    sfs = SequentialFeatureSelector(model).fit(X, y)
    rfe = RFECV(model).fit(X, y)
    return pd.concat([
        pd.Series(sfs.support_, index=X.columns), 
        pd.Series(rfe.support_, index=X.columns), 
        pd.Series(rfe.ranking_, index=X.columns)
    ], axis=1).set_axis(['Seq.Selector', 'Eliminator', 'Eliminator rank'], axis='columns')

def CalcLinRegWithVarSelect(X: pd.DataFrame, y: pd.Series, beta_tolerance=0.05) -> pd.DataFrame: 
    if y.name in X.columns:
        X = X.drop(y.name, axis='columns')
    result = pd.concat([CalcLinearRegression(X, y), VariablesSelection(X, y)], axis=1)
    result['Final Selection'] = (result['coef'] >= 0.05) & (result['P>|t|'] <= 0.05) & result['Seq.Selector']
    return result

def SelectFirstOrderDrivers(X: pd.DataFrame, y: pd.DataFrame, beta_tolerance=0.05): 
    return pd.concat(
        [CalcLinRegWithVarSelect(X, y[col], beta_tolerance=beta_tolerance)['Final Selection'] for col in y.columns], 
        axis=1
    ).any(axis=1)



class ExploratoryAnalysis: 
    #factor_columns = ['@input.Affinity', '@input.MeetNeeds', '@input.Dynamic', '@input.Unique']
    factor_columns = {
        '@input.Affinity': 'EMOTIONAL', 
        '@input.MeetNeeds': 'RATIONAL', 
        '@input.Dynamic': 'LEADERSHIP', 
        '@input.Unique': 'UNIQUENESS',
    }
    #mdf_columns = ['Meaningful', 'Different', 'Salient']
    mdf_columns = {
        'Meaningful': 'GRATIFICATION', 
        'Different': 'DISTINCTION', 
        'Salient': 'AMPLIFICATION',
    }
    #power_premium_columns = ['Power', 'Premium v2']
    power_premium_columns = {
        'Power': 'POWER', 
        'Premium v2': 'PREMIUM'
    }

    worksheet_image = 'image_sets'
    worksheet_data = 'data'

    def __init__(self):
        # refactor to OrderedDict
        self.image_sets = []
        self.image_sets_names = []

    def __ReadImageSets(self): 
        data = pd.read_excel(self.data_file_name, sheet_name=self.worksheet_image, index_col=0)
        for set_ in data.columns: 
            if data[set_].notna().any(): 
                self.image_sets.append(data.loc[data[set_].notna(), set_].to_dict())
                self.image_sets_names.append(set_)

    def __ImageColumnsRenamer(self, image_set: dict) -> dict:
        return {'IMG{:02d}_'.format(k): v for k, v in image_set.items()}
    
    def __SOEColumnsRenamer(self, image_set: dict) -> dict:
        return {'soe.IMG{:02d}_'.format(k): v for k, v in image_set.items()}
    
    def GetCleanSOEDataForSet(self, image_set: dict) -> pd.DataFrame:
        return self.clean_data.rename(columns=self.__SOEColumnsRenamer(image_set))[image_set.values()]
    
    def GetCleanSOEDataForList(self, image_list: list) -> pd.DataFrame:
        return self.clean_data.rename(columns=self.__SOEColumnsRenamer(self.image_sets[0]))[image_list]

    def ReadData(self, data_file_name):
        self.data_file_name = data_file_name
        self.data = pd.read_excel(data_file_name, sheet_name=self.worksheet_data)
        #self.clean_data = self.data[self.data[].notna().all(axis=1)]
        self.clean_data = self.data[self.data['Meaningful'].notna()]

        self.__ReadImageSets()

        # если таких столбцов нет - списки окажутся пустыми 
        self.factor_columns = [c for c in self.factor_columns if c in self.data.columns]
        self.mdf_columns = [c for c in self.mdf_columns if c in self.data.columns]
        self.power_premium_columns = [c for c in self.power_premium_columns if c in self.data.columns]

    def __EquityColumns(self):
        return self.factor_columns + self.mdf_columns + self.power_premium_columns


    def CalcFA(self, df: pd.DataFrame, n_factors):
        fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax').fit(df)
        return pd.DataFrame(fa.loadings_, index=df.columns, columns=['F{}'.format(i+1) for i in range(n_factors)])
    
    def CalcBayesNetworkGraph(self, df: pd.DataFrame):
        hc = HillClimbSearch(df)
        model = hc.estimate(scoring_method=BicScore(df))
        #print(model.edges()) 
        stream = io.BytesIO()
        model.to_graphviz().draw(stream, format='png', prog='dot')
        return stream

    def ReportBayesNetworkGraphs(self): 
        for name, set_ in zip(self.image_sets_names, self.image_sets):
            soe_data = self.GetCleanSOEDataForSet(set_)
            #self.reporter.AddTable(soe_data, 'Graphs_{}'.format(name))
            self.reporter.AddImage(self.CalcBayesNetworkGraph(soe_data), 'Graphs_{}'.format(name))
            self.reporter.AddImage(self.CalcBayesNetworkGraph(soe_data.mul(10).round()), 'Graphs_{}'.format(name), "A30")
    
    def ReportFAs(self): 
        image_set = self.image_sets[0]
        for i in range(3, len(image_set) + 1):
            self.reporter.AddTable(self.CalcFA(self.GetCleanSOEDataForSet(image_set), i), 
                                   'factor_solutions', 
                                   conditional_formatting=True)


    def ReportImageCrossCorrelations(self):
        image_set = self.image_sets[0]
        corr_table = self.GetCleanSOEDataForSet(image_set).corr()
        #corr_table = corr_table.rename(columns=self.soe_columns_renamer, index=self.soe_columns_renamer)
        for r, c in product(range(len(corr_table)), range(len(corr_table))):
            if c >= r:
                corr_table.iloc[r, c] = np.nan
        
        self.reporter.AddTable(corr_table, 'iamge cross correlations', conditional_formatting=True)

    def ReportImageToEquityCorrelations(self):
        if not self.__EquityColumns():
            return
        data = pd.concat([
                self.GetCleanSOEDataForSet(self.image_sets[0]), 
                self.clean_data[self.__EquityColumns()]
            ], axis=1)
        self.reporter.AddTable(
            data.corr().loc[self.image_sets[0].values(), self.__EquityColumns()],
            'image to equity correlations', 
            conditional_formatting=True
        )


    def ReportFirstOrderDrivers(self): 
        for name, set_ in zip(self.image_sets_names, self.image_sets):
            X = Standardize(self.GetCleanSOEDataForSet(set_), across_all_dataframe=True)
            Y = Standardize(self.clean_data[self.factor_columns], across_all_dataframe=False)
            for y_col in Y.columns:    
                y = Y[y_col]
                self.reporter.AddTable(
                    pd.concat([CalcLinearRegression(X, y), VariablesSelection(X, y)], axis=1), 
                    'FOD {}'.format(name)
                )


    
    def ExploratoryReport(self, data_file_name):
        self.reporter = ExcelReportBuilder(data_file_name.split('.')[0] + '_report.xlsx')
        self.ReadData(data_file_name)
        
        self.ReportImageCrossCorrelations()
        self.ReportImageToEquityCorrelations()
        self.ReportFAs()

        self.ReportBayesNetworkGraphs()

        self.ReportFirstOrderDrivers()

        self.reporter.SaveToFile()
        
        