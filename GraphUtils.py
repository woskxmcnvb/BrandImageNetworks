import io
from PIL import Image

import pandas as pd

import pygraphviz as pgv

import plspm.config as c
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode

class PLSPMModel:
    model: Plspm | None
    config: c.Config | None

    def __init__(self, edges=None, scaled=False):
        self.structure = c.Structure()
        self.nodes = []

        self.scaled = scaled

        self.model = None
        self.config = None

        if edges:
            self.ConfigFromEdges(edges)

    def ConfigFromEdges(self, edges: list): 
        assert self.config is None, "Already configured"

        # path model
        for from_, to_ in edges: 
            self.structure.add_path(['lv_{}'.format(from_)], ['lv_{}'.format(to_)])

        self.config = c.Config(self.structure.path(), scaled=self.scaled)

        # measurement model 
        for nodes in edges: 
            for node in nodes:
                if node not in self.nodes:
                    self.config.add_lv('lv_{}'.format(node), Mode.A, c.MV(node))
                    self.nodes.append(node)
        
    def Fit(self, data: pd.DataFrame):
        assert self.config, "Run .ConfgFromEdges first"
        self.model = Plspm(data, self.config, Scheme.CENTROID)
        return self

    def GetPathCoefs(self): 
        assert self.model, "Run .Fit first"
        return self.model.path_coefficients()
    
    def GetEdgeCoef(self, edge: tuple):
        assert self.model, "Run .Fit first"
        from_, to_ = ['lv_{}'.format(node) for node in edge]
        return self.GetPathCoefs().loc[to_, from_]


class GraphManipulations:
    graph: pgv.AGraph | None
    plspm: PLSPMModel | None 

    def __init__(self, graph): 
        if isinstance(graph, pgv.AGraph): 
            self.graph = graph.copy()
        elif isinstance(graph, list): 
            self.graph = pgv.AGraph(directed=True)
            self.graph.add_edges_from(graph)
        else: 
            self.graph = graph.to_graphviz()
        
        self.plspm = None

    def AddEdgeWeight(self, edge, weight): 
        assert self.graph.has_edge(*edge), "No edge " + str(edge)
        e = self.graph.get_edge(*edge)
        e.attr['label'] = '{:.2f}'.format(weight)
    
    def Edges(self):
        return self.graph.edges()
    
    def Plot(self):
        stream = io.BytesIO()
        self.graph.draw(stream, format='png', prog='dot')
        return Image.open(stream)
    
    def FitPLSPM(self, data: pd.DataFrame): 
        # check all nodes in columns
        self.plspm = PLSPMModel(self.Edges()).Fit(data)

    def AppendPLSWeights(self, data):
        self.FitPLSPM(data)
        for edge in self.Edges(): 
            self.AddEdgeWeight(edge, self.plspm.GetEdgeCoef(edge))
