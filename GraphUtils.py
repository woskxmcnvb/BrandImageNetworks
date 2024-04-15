import io
from PIL import Image

#from collections import defaultdict

import pandas as pd

import pygraphviz as pgv

import plspm.config as c
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode





class ModelSpec:
    allowed_types = ['formative', 'reflective']

    class LV: 
        def __init__(self, name: str, type: str, indicators: list | str):
            self.type = type
            self.name = name
            if isinstance(indicators, list):
                self.indicators = indicators.copy()
            elif isinstance(indicators, str):
                self.indicators = [indicators]
            else:
                raise ValueError("Wrong LV definition")

        def __repr__(self):
            return "{} {} {}".format(self.name, self.type, self.indicators)

        def AppendIndicator(self, name: str, type: str, indicator: str):
            assert (self.type == type) and (self.name == name), "Inconsistent LV definition"
            self.indicators.append(indicator)


    edges: list[tuple[str, str]]
    constructs: dict[str, LV]

    def __init__(self, spec: list[dict]) -> None:
        self.edges = list()
        self.constructs = dict()
        self.__AppendFromList(spec)

    def __AppendFromList(self, spec: list[dict]):
        for item in spec: 
            if item['type'] in  self.allowed_types: 
                if item['from'] in self.constructs.keys():
                    self.constructs[item['from']].AppendIndicator(item['from'], item['type'], item['to'])
                else:
                    self.constructs[item['from']] = self.LV(item['from'], item['type'], item['to'])
            elif item['type'] == 'path':
                self.edges.append((item['from'], item['to']))
            else:
                raise ValueError("Wrong LV definition: {}".format(item))


class PLSPMModel:
    model: Plspm | None
    config: c.Config | None

    def Reset(self):
        self.structure = c.Structure()
        self.nodes = list()
        self.model = None
        self.config = None

    def __init__(self, edges=None, scaled=False):
        self.Reset()
        self.scaled = scaled
        if edges:
            self.ConfigFromEdges(edges)

    def ConfigFromEdges(self, edges: list): 
        ##########
        ########## переделать
        ##########
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

        return self

    def ConfigFromModel(self, model: ModelSpec): 
        self.Reset() 
        
        lvs_to_create = set()
        mvs_already_created = set()

        # path model
        for from_, to_ in model.edges: 
            self.structure.add_path(['lv_{}'.format(from_)], ['lv_{}'.format(to_)])
            lvs_to_create.add(from_)  
            lvs_to_create.add(to_)
        
        self.config = c.Config(self.structure.path(), scaled=self.scaled)
        
        # composite latent variables 
        for name, lv in model.constructs.items():
            # inconsistency check
            for i in lv.indicators:
                if i in mvs_already_created:
                    raise ValueError('Problem with {}: double use'.format(i))
            mvs_already_created.update(lv.indicators)
                
            mvs = [c.MV(i) for i in lv.indicators]
            mode = Mode.A if lv.type == 'formative' else 'reflective'
            self.config.add_lv('lv_{}'.format(name), mode, *mvs)
            lvs_to_create.remove(name)

        # the rest - single indicator variables
        for lv in lvs_to_create:
            # inconsistency check 
            if lv in mvs_already_created:
                raise ValueError('Problem with {}: double use'.format(lv))
            mvs_already_created.add(lv)

            self.config.add_lv('lv_{}'.format(lv), Mode.A, c.MV(lv))

        return self
        
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
    model_spec: ModelSpec | None

    def __FromEdges(self, edges: list):
        self.graph = pgv.AGraph(directed=True)
        self.graph.add_edges_from(edges)
    
    def __init__(self, graph): 
        if isinstance(graph, pgv.AGraph): 
            self.graph = graph.copy()
        elif isinstance(graph, list): 
            self.__FromEdges(graph)
            #self.graph = pgv.AGraph(directed=True)
            #self.graph.add_edges_from(graph)
        elif isinstance(graph, ModelSpec):
            self.model_spec = graph
            self.__FromEdges(graph.edges)
        else: 
            self.graph = graph.to_graphviz()
        
        self.plspm = None

    def ResetModels(self):
        self.plspm = None

    def AddEdges(self, edges: list): 
        self.ResetModels()
        self.graph.add_edges_from(edges)

    def AddEdge(self, edge: tuple):
        self.ResetModels()
        self.graph.add_edge(*edge)
    
    def AddEdgeWeight(self, edge, weight): 
        assert self.graph.has_edge(*edge), "No edge " + str(edge)
        e = self.graph.get_edge(*edge)
        e.attr['label'] = '{:.2f}'.format(weight)
        e.attr['penwidth'] = weight * 10
    
    def Edges(self):
        return self.graph.edges()
    
    def Plot(self):
        stream = io.BytesIO()
        self.graph.draw(stream, format='png', prog='dot')
        return Image.open(stream)
    
    def FitPLSPM(self, data: pd.DataFrame): 
        # check all nodes in columns
        # некоторая черезжопность
        if self.model_spec:
            self.plspm = PLSPMModel().ConfigFromModel(self.model_spec).Fit(data)
        else:
            self.plspm = PLSPMModel(self.Edges()).Fit(data)
        return self

    def AppendPLSWeights(self, data):
        self.FitPLSPM(data)
        for edge in self.Edges(): 
            self.AddEdgeWeight(edge, self.plspm.GetEdgeCoef(edge))
        return self
    

    def SubgraphFrom(self, node: str): 

        def _walk_down(graph: pgv.AGraph, node: str, func):
            for e in graph.out_edges(node):
                func(e)
                _walk_down(graph, e[1], func)

        def _walk_up(graph: pgv.AGraph, node: str, func):
            for e in graph.in_edges(node):
                func(e)
                _walk_down(graph, e[0], func)

        def _copy_paste_edge(edge: pgv.agraph.Edge, graph: pgv.AGraph):
            graph.add_edge(*edge, label=edge.attr['label'], penwidth=edge.attr['penwidth'])
        
        sub_graph = pgv.AGraph(directed=True)
        _walk_down(self.graph, node, lambda x: _copy_paste_edge(x, sub_graph))
        _walk_up(self.graph, node,   lambda x: _copy_paste_edge(x, sub_graph))

        key_node = sub_graph.get_node(node)
        key_node.attr['style'] = 'filled'
        key_node.attr['fillcolor'] = 'black'
        key_node.attr['fontcolor'] = 'white'
        key_node.attr['fontsize'] = 24

        return GraphManipulations(sub_graph)
    
    def RouteLenTo(self, node: str):
        total_route_len = 0
        
        def _walk_down(graph: pgv.AGraph, node: str, carry):
            nonlocal total_route_len
            for e in graph.out_edges(node):
                carry_to_pass = carry * float(e.attr['penwidth']) / 10
                if e[1] in ['EMOTIONAL', 'RATIONAL', 'LEADERSHIP', 'UNIQUENESS']: 
                    total_route_len += carry_to_pass
                else:
                    _walk_down(graph, e[1], carry_to_pass)

        _walk_down(self.graph, node, 1)
        return total_route_len
    
    def AllRouteLengthTo(self):
        result = {}
        for node in self.graph.nodes():
            result[node] = self.RouteLenTo(node)
        return result

        
        


