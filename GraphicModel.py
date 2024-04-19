import io
from PIL import Image

import pandas as pd

import pygraphviz as pgv

import plspm.config as c
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode

import utils

from definitions import *


def TupleToEdgeDict(edge: tuple) -> dict:
    # (from, to, type)
    from_, to_, type_ = edge
    return {FROM_KEY: from_, TO_KEY: to_, TYPE_KEY: type_}


def PlotGraph(graph: pgv.AGraph):
    stream = io.BytesIO()
    if not isinstance(graph, pgv.AGraph):
        graph = graph.to_graphviz()
    graph.draw(stream, format='png', prog='dot')
    return Image.open(stream)


class ModelSpec:

    class Edge:
        def __init__(self, from_: str, to_: str, type: str):
            assert type in EDGE_TYPES, "Wrong path type definition: {} to {}".format(from_, to_)
            self.from_ = from_
            self.to_ = to_
            self.type_ = type
        
        def AsTouple(self, include_type=False): 
            if include_type:
                return (self.from_, self.to_, self.type_)
            else:
                return (self.from_, self.to_)
            
        def Type(self):
            return self.type_

    class Construct:
        TYPE_FORMATIVE = 'formative'
        TYPE_REFLECTIVE = 'reflective'
        TYPES = [TYPE_FORMATIVE, TYPE_REFLECTIVE]
        def __init__(self, name: str, type: str, indicators: list | str):
            self.type_ = type
            self.name_ = name
            self.indicators_ = set()
            if isinstance(indicators, list) or isinstance(indicators, set):
                self.indicators_.update(indicators)
            elif isinstance(indicators, str):
                self.indicators_.add(indicators)
            else:
                raise ValueError("Wrong LV definition {}".format(name))
            
        def Indicators(self):
            return self.indicators_

        def __repr__(self):
            return "{} {} {}".format(self.name_, self.type_, self.indicators_)

        def AppendIndicator(self, name: str, type: str, indicator: str):
            assert isinstance(indicator, str)
            assert (self.type_ == type) and (self.name_ == name), "Inconsistent LV definition {} {} {}".format(name, type, indicator)
            self.indicators_.add(indicator)

    class Node: 
        name: str
        in_edges: set[str]
        out_edges: set[str]
        def __init__(self, name: str):
            self.name = name
            self.in_edges = set()
            self.out_edges = set()
        
        def AddInEdge(self, from_node: str):
            self.in_edges.add(from_node)
        
        def AddOutEdge(self, to_node: str):
            self.out_edges.add(to_node)
        
        def InEdges(self): 
            return self.in_edges
        
        def OutEdges(self):
            return self.out_edges

    edges: list[Edge]
    constructs: dict[str, Construct]
    nodes: dict[str, Node]

    def __init__(self, spec: list | pd.DataFrame | pgv.AGraph = None) -> None:
        self.edges = list()
        self.constructs = dict()
        self.nodes = dict()
        if spec is not None:
            self.AppendEdges(spec)

    def __FromList(self, spec: list[dict]):
        # private потому что нет consistency check
        for item in spec: 
            from_, to_, type_ = item[FROM_KEY], item[TO_KEY], item[TYPE_KEY]
            if type_ in self.Construct.TYPES:
                if self.HasConstruct(from_):
                    self.constructs[from_].AppendIndicator(from_, type_, to_)
                else:
                    self.constructs[from_] = self.Construct(from_, type_, to_)
            elif type_ in EDGE_TYPES:
                self.edges.append(self.Edge(from_, to_, type_))
                self.__AddNode(from_)
                self.__AddNode(to_)
                self.nodes[to_].AddInEdge(from_)
                self.nodes[from_].AddOutEdge(to_)
            else:
                raise ValueError("Wrong LV definition: {}".format(item))
    
    def __FromDF(self, spec: pd.DataFrame):
        # private потому что нет consistency check
        self.__FromList(spec.to_dict('records'))

    def __FromAGraph(self, graph: pgv.AGraph):
        # private потому что нет consistency check
        self.__FromList(
            [{FROM_KEY: from_, TO_KEY: to_, TYPE_KEY: 'path'} for from_, to_ in graph.edges()]
        )
    
    def AppendEdges(self, spec: list | pd.DataFrame | pgv.AGraph) -> None:
        # dict format: {'from': str, 'to': str, 'type': str}
        if isinstance(spec, pd.DataFrame):
            self.__FromDF(spec)
        elif isinstance(spec, list):
            self.__FromList(spec)
        elif isinstance(spec, pgv.AGraph):
            self.__FromAGraph(spec)
        else: 
            raise ValueError("GraphicModel: unknown input to ctor")
        self.ConsistencyCheck()

    def Edges(self, include_type=False) -> list[tuple]:
        return [e.AsTouple(include_type=include_type) for e in self.edges]
    
    def ToDF(self):
        return pd.DataFrame(self.Edges(include_type=True), columns=[FROM_KEY, TO_KEY, TYPE_KEY])
    
    def Nodes(self) -> list[str]:
        return list(self.nodes.keys()) 
    
    def HasConstruct(self, name: str) -> bool:
        return name in self.constructs.keys()
    
    def InEdges(self, node) -> set[str]:
        return self.nodes[node].InEdges()
    
    def OutEdges(self, node) -> set[str]:
        return self.nodes[node].OutEdges()

    def __AddNode(self, name: str):
        if not name in self.nodes.keys():
            self.nodes[name] = self.Node(name)

    def ConsistencyCheck(self):
        all_mvs_for_constructs = set()
        for c in self.constructs.values():
            doubles = all_mvs_for_constructs.intersection(c.indicators_)
            if doubles:
                raise ValueError("Inconsistent model spec: Double use of indicators in constructs {}".format(doubles))
            all_mvs_for_constructs.update(c.indicators_)
        
        all_mvs_in_single_lvs = set()
        for e in self.Edges(include_type=False):
            all_mvs_in_single_lvs.update(e)
        doubles = all_mvs_for_constructs.intersection(all_mvs_in_single_lvs)
        if doubles:
            raise ValueError("Inconsistent model spec: The same indicators in single and constructs {}".format(doubles))



class PLSPMModel:
    model: Plspm | None = None
    config: c.Config | None = None

    def __init__(self, scaled=False):
        self.nodes = list()
        self.scaled = scaled

    def ConfigFromModel(self, model: ModelSpec): 
        
        lvs_to_create = set()

        # path model
        self.structure = c.Structure()
        for from_, to_ in model.Edges(include_type=False): 
            self.structure.add_path(['lv_{}'.format(from_)], ['lv_{}'.format(to_)])
            lvs_to_create.add(from_)  
            lvs_to_create.add(to_)
        
        self.config = c.Config(self.structure.path(), scaled=self.scaled)
        
        # composite latent variables 
        for name, lv in model.constructs.items():
            mvs = [c.MV(i) for i in lv.Indicators()]
            mode = Mode.A if lv.type_ == ModelSpec.Construct.TYPE_FORMATIVE else Mode.B
            self.config.add_lv('lv_{}'.format(name), mode, *mvs)
            lvs_to_create.remove(name)

        # the rest - single indicator variables
        for lv in lvs_to_create:
            self.config.add_lv('lv_{}'.format(lv), Mode.A, c.MV(lv))

        return self
        
    def Fit(self, data: pd.DataFrame):
        assert self.config, "Run .ConfgFromEdges first"
        self.model = Plspm(data, self.config, Scheme.CENTROID)
        return self

    def GetPathCoefs(self): 
        assert self.model, "Run .Fit first"
        return self.model.path_coefficients()
    
    def GetPathCoef(self, edge: tuple):
        edge = edge[:2]
        assert self.model, "Run .Fit first"
        from_, to_ = ['lv_{}'.format(node) for node in edge]
        return self.GetPathCoefs().loc[to_, from_]


class GraphicModel:
    # интерфейс для работы с графическими моделями
    # (1) хранит спецификацию модели в классе ModelSpec
    #     инициализируется спецификацией модели в различных форматах 
    # (2) запускает и хранит обсчет модели PLS
    #     SEM пока не реализован 
    # (3) строит графы в различных разрезах в формате pgv.AGraph

    plspm: PLSPMModel | None = None
    model_spec: ModelSpec | None = None

    def __init__(self, spec=None): 
        self.model_spec = ModelSpec(spec)

    def ResetModels(self):
        self.plspm = None

    def AppendMDFGraph(self):
        self.AppendEdges(MDF_GRAPH)

    def AppendEdges(self, spec):
        self.ResetModels()
        self.model_spec.AppendEdges(spec)
  
    def FitPLSPM(self, data: pd.DataFrame): 
        self.plspm = PLSPMModel().ConfigFromModel(self.model_spec).Fit(data)
        return self

    def Graph(self, add_pls_weights=True, exclude_mdf=True):
        if add_pls_weights and (self.plspm is None):
            print("Warning! PLS is not fit, building without PLS weights")
            add_pls_weights = False
        
        graph = pgv.AGraph(directed=True)
        for edge in self.model_spec.Edges(include_type=True):
            if exclude_mdf and edge[2] == EDGE_TYPE_MDF:
                continue
            if add_pls_weights: 
                wt = self.plspm.GetPathCoef(edge)
                graph.add_edge(*edge[:2], label='{:.2f}'.format(wt), penwidth=wt * 10)
            else:
                graph.add_edge(*edge[:2])
        
        return graph
    
    def SubgraphFromNodes(self, node: str | list[str], add_pls_weights=True, exclude_mdf=True): 
        
        def _walk_down(from_: str, func):
            for to_ in self.model_spec.OutEdges(from_):
                func((from_, to_))
                _walk_down(to_, func)

        def _walk_up(to_: str, func):
            for from_ in self.model_spec.InEdges(to_):
                func((from_, to_))
                _walk_up(from_, func)

        def _add_edge(edge: tuple[str], graph: pgv.AGraph):
            if exclude_mdf and ((edge[0] in MDFS_POWER_PREMIUM) or (edge[1] in MDFS_POWER_PREMIUM)):
                return
            if add_pls_weights and self.plspm:
                wt = self.plspm.GetPathCoef(edge)
                graph.add_edge(*edge, label='{:.2f}'.format(wt), penwidth=wt * 10)
            else:
                graph.add_edge(*edge)
        
        if isinstance(node, str): 
            node = [node]
        elif not isinstance(node, list):
            raise ValueError(".SubgraphFromNode: Wrong nodes definition {}".format(node))
        
        sub_graph = pgv.AGraph(directed=True)
        
        for n in node: 
            _walk_down(n, lambda x: _add_edge(x, sub_graph))
            _walk_up(  n, lambda x: _add_edge(x, sub_graph))

            key_node = sub_graph.get_node(n)
            key_node.attr['style'] = 'filled'
            key_node.attr['fillcolor'] = 'black'
            key_node.attr['fontcolor'] = 'white'
            key_node.attr['fontsize'] = 24

        return sub_graph
    
    def PathLen(self, start_node: str, end_nodes: list[str] | str):
        assert self.plspm, "Model not fit. Run .Fit... first"
        if isinstance(end_nodes, str): 
            end_nodes = [end_nodes]
        elif not isinstance(end_nodes, list):
            raise ValueError(".RouteLen: Wrong to_ definition")
        
        total_route_len = 0
        
        def _walk_down(from_: str, carry):
            nonlocal total_route_len
            for to_ in self.model_spec.OutEdges(from_):
                carry_to_pass = carry * self.plspm.GetPathCoef((from_, to_))
                if to_ in end_nodes: 
                    total_route_len += carry_to_pass
                else:
                    _walk_down(to_, carry_to_pass)

        _walk_down(start_node, 1)
        return total_route_len
    
    def PathLenFromAllNodes(self, end_nodes: list[str] | str):
        if isinstance(end_nodes, str): 
            end_nodes = [end_nodes]
        elif not isinstance(end_nodes, list):
            raise ValueError(".RouteLen: Wrong to_ definition")
        
        assert utils.AllElementsAreThere(end_nodes, self.model_spec.Nodes()), "Not all nodes are present in the model"
        
        result = {}
        for node in self.model_spec.Nodes():
            result[node] = self.PathLen(node, end_nodes)
        return pd.Series(result).sort_values(ascending=False)

    