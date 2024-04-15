import io 
from collections import OrderedDict

import pandas as pd
import numpy as np
from itertools import product

from ExcelReportBuilder import ExcelReportBuilder
from GraphUtils import GraphManipulations

from factor_analyzer import FactorAnalyzer

from pgmpy.estimators import HillClimbSearch, TreeSearch
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

def CalcBayesNetwork(df: pd.DataFrame):
    return HillClimbSearch(df).estimate(scoring_method=BicScore(df))

def CalcTree(df: pd.DataFrame, root_node: str): 
    return TreeSearch(df, root_node=root_node).estimate(estimator_type='chow-liu')


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

    mdf_graph = [
        ('EMOTIONAL', 'GRATIFICATION'),
        ('RATIONAL', 'GRATIFICATION'),
        ('LEADERSHIP', 'GRATIFICATION'),
        ('UNIQUENESS', 'GRATIFICATION'),
        ('EMOTIONAL', 'DISTINCTION'),
        ('RATIONAL', 'DISTINCTION'),
        ('LEADERSHIP', 'DISTINCTION'),
        ('UNIQUENESS', 'DISTINCTION'),
        ('EMOTIONAL', 'AMPLIFICATION'),
        ('RATIONAL', 'AMPLIFICATION'),
        ('LEADERSHIP', 'AMPLIFICATION'),
        ('UNIQUENESS', 'AMPLIFICATION'),
        ('GRATIFICATION', 'POWER'),
        ('DISTINCTION', 'POWER'),
        ('AMPLIFICATION', 'POWER')]

    image_sets = ExtOrderedDict()
    graphs_for_sets: dict[str: GraphManipulations] = {}

    worksheet_image = 'image_sets'
    worksheet_data = 'data'
    worksheet_model = 'model_spec'

    

    def __init__(self):
        self.image_sets.clear()
        self.graphs_for_sets.clear()

    def __EquityColumns(self):
        return self.factor_columns + self.mdf_columns + self.power_premium_columns
    
    def __ReadModelSpec(self): 
        self.image_sets.clear()
        self.graphs_for_sets.clear()
        data = pd.read_excel(self.data_file_name, sheet_name=self.worksheet_image, index_col=0)
        for set_ in data.columns: 
            if data[set_].notna().any(): 
                self.image_sets[set_] = data[set_].dropna().to_dict()
                self.graphs_for_sets[set_] = None # will GraphManipulation object

    def __ReadImageSets(self): 
        self.image_sets.clear()
        self.graphs_for_sets.clear()
        data = pd.read_excel(self.data_file_name, sheet_name=self.worksheet_image, index_col=0)
        for set_ in data.columns: 
            if data[set_].notna().any(): 
                self.image_sets[set_] = data[set_].dropna().to_dict()
                self.graphs_for_sets[set_] = None # will GraphManipulation object


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

    def CleanSOEandMDFData(self): 
        return pd.concat([self.clean_mdf_data, self.clean_soe_data], axis=1)

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
        for set_name, set_ in self.image_sets.items():
            G1 = GraphManipulations(CalcBayesNetwork(self.clean_soe_data[set_.values()]))
            self.reporter.AddImage(G1.Plot(), 'Graphs_{}'.format(set_name))
            
            self.graphs_for_sets[set_name] = GraphManipulations(CalcBayesNetwork(self.clean_soe_data[set_.values()].mul(10).round()))
            self.reporter.AddImage(self.graphs_for_sets[set_name].Plot(), 'Graphs_{}'.format(set_name), "AA1")
        print("Bayes nets graphs - Ok")
    

    def ReportFirstOrderDrivers(self): 
        Y = Standardize(self.clean_mdf_data, across_all_dataframe=False)
        XX = Standardize(self.clean_soe_data, across_all_dataframe=True)
        for set_name, set_ in self.image_sets.items():
            X = XX[set_.values()]
            for target_name, y in Y.items():
                fod = CalcLinRegWithVarSelect(X, y)
                self.reporter.AddTable(fod, 'FOD {}'.format(set_name))
                if target_name in self.factor_columns:
                    self.graphs_for_sets[set_name].AddEdges([(ix, target_name) for ix, val in fod['Final Selection'].items() if val])
        print("First Order Drivers - Ok")

    def ReportGraphs(self):
        self.ReportBayesNetworkGraphs()
        self.ReportFirstOrderDrivers() 
        data = self.CleanSOEandMDFData()
        for set_name, set_ in self.image_sets.items():
            self.graphs_for_sets[set_name].AddEdges(self.mdf_graph)
            self.graphs_for_sets[set_name].AppendPLSWeights(data) 
            self.reporter.AddImage(self.graphs_for_sets[set_name].Plot(), 'BSA_{}'.format(set_name))

    def ReportTreeGraphs(self):
        for col in self.clean_soe_data: 
            self.reporter.AddImage(
                GraphManipulations(CalcTree(self.clean_soe_data, col)).Plot(), 
                'TREE {}'.format(col)
                )
    
    def ExploratoryReport(self, data_file_name):
        self.reporter = ExcelReportBuilder(data_file_name.split('.')[0] + '_report.xlsx')
        self.ReadDataFile(data_file_name)
        
        self.ReportImageCrossCorrelations()
        self.ReportFAs()
        self.ReportSOEToEquityCorrelations()
        
        self.ReportGraphs()
        self.ReportTreeGraphs()

        self.reporter.SaveToFile()
        
        