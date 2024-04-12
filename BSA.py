import io 
from collections import OrderedDict

import pandas as pd
import numpy as np
from itertools import product


from ExcelReportBuilder import ExcelReportBuilder
from factor_analyzer import FactorAnalyzer

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore

import statsmodels.api as sm
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression


class ExtOrderedDict(OrderedDict):
    def top(self):
        return next(iter(self.items()))


def Standardize(df, across_all_dataframe=False): 
    if across_all_dataframe: 
        return df.sub(df.mean(axis=None)).div(df.stack().std())
    else:
        return df.sub(df.mean()).div(df.std())
    

def CalcFA(df: pd.DataFrame, n_factors) -> pd.DataFrame:
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax').fit(df)
    return pd.DataFrame(fa.loadings_, index=df.columns, columns=['F{}'.format(i+1) for i in range(n_factors)])


def CalcBayesNetworkGraph(df: pd.DataFrame) -> io.BytesIO:
    hc = HillClimbSearch(df)
    model = hc.estimate(scoring_method=BicScore(df))
    #print(model.edges()) 
    stream = io.BytesIO()
    model.to_graphviz().draw(stream, format='png', prog='dot')
    return stream


##################################################
############# First Order Dirvers ################
##################################################

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
    result['Final Selection'] = (result['coef'] >= beta_tolerance) & (result['P>|t|'] <= 0.05) & result['Seq.Selector']
    return result

def SelectFirstOrderDrivers(X: pd.DataFrame, y: pd.DataFrame, beta_tolerance=0.05): 
    return pd.concat(
        [CalcLinRegWithVarSelect(X, y[col], beta_tolerance=beta_tolerance)['Final Selection'] for col in y.columns], 
        axis=1
    ).any(axis=1)

##################################################
##################################################


def CheckColumnsPresent(data:pd.DataFrame, cols: list) -> bool:
    return all([col in data.columns for col in cols])



class ExploratoryAnalysis: 
    factor_columns = ['EMOTIONAL', 'RATIONAL', 'LEADERSHIP', 'UNIQUENESS']
    mdf_columns = ['GRATIFICATION', 'DISTINCTION', 'AMPLIFICATION']
    power_premium_columns = ['POWER', 'PREMIUM']

    image_sets = ExtOrderedDict()

    worksheet_image = 'image_sets'
    worksheet_data = 'data'

    def __init__(self):
        self.image_sets.clear()

    def __EquityColumns(self):
        return self.factor_columns + self.mdf_columns + self.power_premium_columns

    def __ReadImageSets(self): 
        self.image_sets.clear()
        data = pd.read_excel(self.data_file_name, sheet_name=self.worksheet_image, index_col=0)
        for set_ in data.columns: 
            if data[set_].notna().any(): 
                self.image_sets[set_] = data[set_].dropna().to_dict()

    def FullImageList(self):
        return self.image_sets.top()[1]

    """def __ImageColumnsRenamer(self, image_set: dict) -> dict:
        return {'IMG{:02d}_'.format(k): v for k, v in image_set.items()}
    
    def __SOEColumnsRenamer(self, image_set: dict) -> dict:
        return {'soe.IMG{:02d}_'.format(k): v for k, v in image_set.items()}"""
    
    """def GetCleanSOEDataForSet(self, image_set: dict) -> pd.DataFrame:
        return self.clean_data.rename(columns=self.__SOEColumnsRenamer(image_set))[image_set.values()]
    
    def GetCleanSOEDataForList(self, image_list: list) -> pd.DataFrame:
        return self.clean_data.rename(columns=self.__SOEColumnsRenamer(self.FullImageList()))[image_list]"""
    
    def __ReadData(self): 
        data = pd.read_excel(self.data_file_name, sheet_name=self.worksheet_data)

        # dropna
        img_cols = ['IMG{:02d}_'.format(k) for k in self.FullImageList().keys()]
        data = data[data[img_cols].notna().all(axis=1)]
        
        soe_renamer = {'soe.IMG{:02d}_'.format(k): v for k, v in self.FullImageList().items()}
        if not CheckColumnsPresent(data, soe_renamer.keys()):
            raise ValueError("Not all SOE Images listed in first set are present in the data")
        self.clean_soe_data = data[soe_renamer.keys()].rename(columns=soe_renamer)

        # not used yet 
        #img_renamer = {'IMG{:02d}_'.format(k): v for k, v in self.FullImageList().items()}
        #self.clean_img_data = data.rename(soe_renamer)

        mdf_renamer = {
            '@input.Affinity': 'EMOTIONAL', 
            '@input.MeetNeeds': 'RATIONAL', 
            '@input.Dynamic': 'LEADERSHIP', 
            '@input.Unique': 'UNIQUENESS',
            'Meaningful': 'GRATIFICATION', 
            'Different': 'DISTINCTION', 
            'Salient': 'AMPLIFICATION',
            'Power': 'POWER', 
            'Premium v2': 'PREMIUM'}
        if not CheckColumnsPresent(data, mdf_renamer.keys()):
            raise ValueError("Not all MDF variables are present in the data. Expected: " + str(mdf_renamer.keys()))
        self.clean_mdf_data = data[mdf_renamer.keys()].rename(columns=mdf_renamer)

    def ReadDataFile(self, data_file_name):
        self.data_file_name = data_file_name
        self.__ReadImageSets()
        self.__ReadData()
        print("Read File - Ok")


    def ReportImageCrossCorrelations(self):
        corr_table = self.clean_soe_data.corr()
        # clean diagonal
        for r, c in product(range(len(corr_table)), range(len(corr_table))):
            if c == r:
                corr_table.iloc[r, c] = np.nan
        self.reporter.AddTable(corr_table, 'SOE cross-correlations', conditional_formatting=True)
        print("SOE cross-correlations - Ok")


    def ReportFAs(self): 
        for i in range(3, len(self.FullImageList()) + 1):
            self.reporter.AddTable(CalcFA(self.clean_soe_data, i), 
                                   'factor_solutions', 
                                   conditional_formatting=True)
        print("SOE FAs - Ok")
    
    def ReportSOEToEquityCorrelations(self):
        data = pd.concat([self.clean_mdf_data, self.clean_soe_data], axis=1) 
        self.reporter.AddTable(
            data.corr().loc[self.clean_soe_data.columns, self.clean_mdf_data.columns],
            'SOE to equity correlations', 
            conditional_formatting=True
        )
        print("SOE to equity correlations - Ok")

    def ReportBayesNetworkGraphs(self): 
        for name, set_ in self.image_sets.items():
            self.reporter.AddImage(CalcBayesNetworkGraph(self.clean_soe_data[set_.values()]), 'Graphs_{}'.format(name))
            self.reporter.AddImage(CalcBayesNetworkGraph(self.clean_soe_data[set_.values()].mul(10).round()), 'Graphs_{}'.format(name), "D5")
        print("Bayes nets graphs - Ok")
    

    def ReportFirstOrderDrivers(self): 
        Y = Standardize(self.clean_mdf_data, across_all_dataframe=False)
        XX = Standardize(self.clean_soe_data, across_all_dataframe=True)
        for set_name, set_ in self.image_sets.items():
            X = XX[set_.values()]
            for target_name, y in Y.items():
                self.reporter.AddTable(
                    CalcLinRegWithVarSelect(X, y), 'FOD {}'.format(set_name)
                )
        print("First Order Drivers - Ok")


    
    def ExploratoryReport(self, data_file_name):
        self.reporter = ExcelReportBuilder(data_file_name.split('.')[0] + '_report.xlsx')
        self.ReadDataFile(data_file_name)
        
        #self.ReportImageCrossCorrelations()
        #self.ReportFAs()
        #self.ReportSOEToEquityCorrelations()
        #self.ReportBayesNetworkGraphs()
        self.ReportFirstOrderDrivers()

        self.reporter.SaveToFile()
        
        