import time
import os
from itertools import product
import itertools
import math

import networkx as nx
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from stellargraph.data import UniformRandomWalk, BiasedRandomWalk, UniformRandomMetaPathWalk
from stellargraph import StellarGraph, StellarDiGraph, datasets
from neo4j import GraphDatabase, unit_of_work


@unit_of_work(timeout=300)
def getSubgraph_neo4j_COP(graph_uri, node_name, query, node_labels, 
    compared_labels = None):
    """Takes in a targeted subgraph, a specific node name, a CYPHER query, 
    and a list of valid labels (nodes can have multiple labels, this helps us
    keep the most important ones). For the specific node provided, this queries
    the graph and returns the subgraph.

    Parameters
    ----------
    graph_uri : string 
        A bolt address for a Neo4J server. 
        This will be connected to by the neo4j GraphDatabase object..

    node_name : string 
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    query : string 
        A CYPHER query we use to query the graph and build the subgraph.
        This string should have a %s character which will be substitued 
        with the provided **node_name**. This is because most workflows
        center around the use of automated queries and a varied list of
        node names.

    node_labels : array-like
        Used in determining which node labels should be kept as primary
        identifiers when nodes have multiple labels.
    
    compare_labels : array-like (default: None)
        Used in determining which node labels should be kept as primary
        identifiers when nodes have multiple labels. This has a lower
        priority than the **node_labels** parameter.
    
    Returns
    -------
    sg : StellarDiGraph  
        A directional graph object representing the node neighborhood.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spectral_layout(G)

    Notes
    -----
    """
    
    queryStr = query % node_name   
    
    driver = GraphDatabase.driver(graph_uri)

    user_labels = list(set(node_labels))
            
    with driver.session() as session:
        #print(queryStr)
        result = session.run(queryStr)
        d = {}
        join_values = []
        for node in result.graph().nodes:
            node_name = node['name']
            if node_name not in join_values:
                #print('labels = ',list(i.labels))
                if len(node.labels)>1:
                    for m in node.labels:
                        if m in user_labels:
                            node_type = m
                        
                        ### for multiple-labeled graph using regex "? ? ?"
                        
                        elif compared_labels != None:
                            if m in compared_labels:
                                node_type = m
                        else:
                            node_type = list(node.labels)[0]
                        ###
                else:
                    node_type = list(node.labels)[0]
                s = d.get(node_type,set())
                s.add(node_name)
                d[node_type] = s
            join_values.append(node_name)

        rels = set()
        for rel in result.graph().relationships:
            start = rel.start_node["name"]
            end = rel.end_node["name"]
            rel_type = rel.type
            rels.add((start, end, rel_type))

    raw_nodes = d        
    edges = pd.DataFrame.from_records(list(rels),columns=["source","target","label"])

    data_frames = {}

    #For each node_label *k* create a dictonary.
    for k in d:
        node_names = list(d[k])
        df = pd.DataFrame({"name":node_names}).set_index("name")
        data_frames[k] = df
    #print(edges)
    sg = StellarDiGraph(data_frames,edges=edges, edge_type_column="label")
   
    return sg 

def buildCypher(s,t,k_nodes,k_val):
    query = "MATCH "
    query += "p1=(s:`%s`)" % s
    for k in range(k_val):
        query += "--(x%i)" % (k+1)
    if(t!=None): query+= "--(t:`%s`)" % t

    query += ' WHERE s.name="%s" '
    label_clauses = []
    #For each k_level we produce a query string which looks like
    #x1:`gene` OR x1:`drug`
    for k in range(k_val):
        if(k_nodes[k]==None or len(k_nodes[k])==0):continue
        #where_str += ( ) 
        clauses = ["x%i:`%s`" % (k+1,label) for label in k_nodes[k]]
        clause_str = " ( " + " OR ".join(clauses) + " ) "
        label_clauses.append(clause_str)
    if(len(label_clauses)!=0):
        #Add and AND clause because we start the WHERE s.name="XYZ" AND 
        query += " AND "
        query += " AND ".join(label_clauses)
    #query += " RETURN * " 

    #query += " WITH collect(p1) as nodez UNWIND nodez as c RETURN c "
    query += " RETURN p1 LIMIT 5000 "
    return query

def buildCypherNodesAndEdges(s,t,t_edges,k_nodes,k_edges,k_val):
    query = "MATCH "
    query += "p1=(s:`%s`)" % s
    for k in range(k_val):
        query += "-[r%i]-(x%i)" % (k+1,k+1)
    #if(t!=None): query+= "--(t:`%s`)" % t
    if(t!=None): query+= "-[rt]-(t:`%s`)" % t

    query += ' WHERE s.name="%s" '
    label_clauses = []
    for k in range(k_val):
        if(k_nodes[k]==None or len(k_nodes[k])==0):continue
        #where_str += ( ) 
        clauses = ["x%i:`%s`" % (k+1,label) for label in k_nodes[k]]
        clause_str = " ( " + " OR ".join(clauses) + " ) "
        label_clauses.append(clause_str)
    for k in range(k_val):
        if(k_edges[k]==None or len(k_edges[k])==0):continue
        #where_str += ( ) 
#        clauses = ["r%i:`%s`" % (k+1,label) for label in k_edges[k]]
        clause_str = "TYPE(r%s) IN [" % (k+1)
        clause_str += ' ,'.join(['"' + x + '"' for x in k_edges[k]])
        clause_str += " ]"
        label_clauses.append(clause_str)
    if(t!=None and  len(t_edges)!=0):
        clause_str = "TYPE(rt) IN [" 
        clause_str += ' ,'.join(['"' + x + '"' for x in t_edges])
        clause_str += " ]"
        #clauses = ["rt:`%s`" % (label) for label in t_edges]
        #clause_str = " ( " + " OR ".join(clauses) + " ) "
        label_clauses.append(clause_str)
#        clause_str = " ( " + " OR ".join(clauses) + " ) "
    if(len(label_clauses)!=0):
    #    query += " WHERE "
        query += " AND "
        query += " AND ".join(label_clauses)
    query += " RETURN p1 LIMIT 5000 "
    return query

def buildSubgraphDictonaryForNodes(graph_uri, node_list, neo4j_query, node_labels, compared_labels=None, debug=False):
    if(debug):print("Building out subgraphs from query")
    subGs = {}
    for node_name in node_list:
        if(debug):print("Build subgraph for ==%s==" % node_name)
        subG = getSubgraph_neo4j_COP(graph_uri, node_name, neo4j_query, node_labels, compared_labels)
        if(len(subG.nodes())==0): subG = None
        subGs[node_name] = subG 
    return subGs

def generateRandomWalks(subgraph_dict, node_list, method, walk_length, num_walks, metapath = None):
    """Takes in a dictonary of subgraphs, a list of nodes you want to walk,
    the method, l, and r.

    Parameters
    ----------
    subgraph_dict : dictonary of StellarDiGraphs 
        A dictonary of graph objects created around the nodes of interest.
        Should contain a key for each string in node_list. See 
        *buildSubgraphDictonary*.

    node_list : array-list of strings
        A list of nodes which to be used as the centers for the walks.
        

    method : string: One of "deepwalk", "node2vec", or "metapath2vec".
        A string which specifies which method to use to generate the 
        random walks.

    walk_length : integer
        A number which determines how long each random walks through the 
        graph should be.
    
    num_walks : integer 
        A number which sets the number of walks to generate for each node
        in **node_list**.
    
    Returns
    -------
    Walks : list of walks 
          A list of walks around each node in **node_list** in it's 
          respective subgraph (from **subgraph_dict[node]**). There
          will be **num_walks** X len(node_list) walks in this list.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spectral_layout(G)

    Notes
    -----
    """
    subGs = {}


    Walks = []
    for node in node_list:
        subG = subgraph_dict[node]
        if(subG == None):continue
        # DeepWalk
        if method == 'deepwalk':
            rw = UniformRandomWalk(subG) #BiasedRandomWalk(G)
            walks = rw.run(
                nodes= [node],
                length = walk_length,
                n = num_walks 
                #seed = 1
            )

        # Node2Vec
        elif method == 'node2vec':
            rw = BiasedRandomWalk(subG)
            walks = rw.run(
                nodes= [node],  
                length = walk_length,  
                n = num_walks,  # number of random walks per root node
                p = 0.25,  # Defines (unormalised) probability, 1/p, of returning to source node
                q = 0.25#,  # Defines (unormalised) probability, 1/q, for moving away from source node
                #seed = 5
            )

        #Metapath2vec
        elif method == 'metapath2vec':
            rw = UniformRandomMetaPathWalk(subG)
            walks = rw.run(
                nodes= [node],#list(G.nodes()),
                length = walk_length,  # maximum length of a random walk
                n = num_walks,  # number of random walks per root node
                metapaths = metapath
                #seed = 5
            )
        else:
            print('Method is not one of "deepwalk", "node2vec", or "metapath2vec".')
            return

        # append walks
        for w in walks:
            Walks.append(w)
            
    return Walks

#Generates a model with embeddings from provided collection of Walks.
def buildModel(Walks):
    str_walks = [[str(n) for n in walk] for walk in Walks]
    #model = Word2Vec(str_walks, vector_size=128, window=10, min_count=0, sg=1, workers=2, iter=5)
    model = Word2Vec(str_walks, vector_size=128, window=10, min_count=0, sg=1, workers=2, epochs=5)
    return model

#Computes various benchmarks for a machine learning models.
def evaluate(model, subgraph_dict, node_list1, node_list2, print_info=True):
    evaluate_dict = {}
    hit_at_1_in_list = 0
    hit_at_3_in_list = 0
    hit_at_5_in_list = 0
    mrr_in_list = 0
    info_tuples = []
    Node_List = node_list1 + node_list2
    for idx, node in enumerate(node_list1):
            #if i[0] == node_list2[node_list1.index(n)]:

        
        pair_node = node_list2[idx]

        print("==", node, "==")
        n_all = 0
        num_in_list = 0
        rank_in_list = 9999 

        #In this loop we step through all of the most similar nodes for our , we seek to find
        # where the pair for the currently selected node is in the list of all similar nodes.
        # For this we walk through all the similar nodes, each time we see a node in our global, 
        # list of all nodes, we tick up a counter. When we find the location of the pair, we terminate. 
        try:
            l = model.wv.most_similar(node, topn = 1)
        except:
            info_tuples.append((node, pair_node, "Cannot find",len(Node_List)))
            continue

        for i in model.wv.most_similar(node, topn = 20000):
            n_all += 1

            #for j in Node_List:
            #    if i[0] in subgraph_dict[j].nodes():
            #        nodeType = subgraph_dict[j].node_type(i[0])
            #        break

            
            if i[0] in Node_List:
                num_in_list += 1

            #if i[0] == node_list2[node_list1.index(n)]:
            if i[0] == pair_node:
                print('drugs* ',num_in_list)
                # test: include only drugs in list
                rank_in_list = num_in_list
                if rank_in_list == 1:
                    hit_at_1_in_list += 1
                if rank_in_list <= 3:
                    hit_at_3_in_list += 1
                if rank_in_list <= 5:
                    hit_at_5_in_list += 1
                mrr_in_list += 1/rank_in_list
                info_tuples.append((node, pair_node, rank_in_list,len(Node_List)))
                break

    if(print_info):print('compute only in list:')
    if(print_info):print("num of Compound*: ",num_in_list)
    if(print_info):print("HIT@1 = ", round(hit_at_1_in_list/len(node_list1),4))
    if(print_info):print("HIT@3 = ", round(hit_at_3_in_list/len(node_list1),4))
    if(print_info):print("HIT@5 = ", round(hit_at_5_in_list/len(node_list1),4))
    if(print_info):print("MRR = ", round(mrr_in_list/len(node_list1),4))
    HIT1 = round(hit_at_1_in_list/len(node_list1),4)
    HIT3 = round(hit_at_3_in_list/len(node_list1),4)
    HIT5 = round(hit_at_5_in_list/len(node_list1),4)
    MRR = round(mrr_in_list/len(node_list1),4)
    
    evaluate_dict['HIT@1'] = HIT1
    evaluate_dict['HIT@3'] = HIT3
    evaluate_dict['HIT@5'] = HIT5
    evaluate_dict['MRR'] = MRR
    
    return evaluate_dict, info_tuples

def evaluate_v2(model, subgraph_dict, node_list1, node_list2, print_info=True):
    info_tuples = []
    Node_List = node_list1 + node_list2
    performance_dict = {}
    for idx, node in enumerate(node_list1):
            #if i[0] == node_list2[node_list1.index(n)]:

        
        pair_node = node_list2[idx]

        print("==", node, "==")
        n_all = 0
        num_in_list = 0
        rank_in_list = 9999 

        #In this loop we step through all of the most similar nodes for our , we seek to find
        # where the pair for the currently selected node is in the list of all similar nodes.
        # For this we walk through all the similar nodes, each time we see a node in our global, 
        # list of all nodes, we tick up a counter. When we find the location of the pair, we terminate. 
        try:
            l = model.wv.most_similar(node, topn = 1)
        except:
            info_tuples.append((node, pair_node, "Cannot find",len(Node_List)))
            continue

        hit_list = []
        for i in model.wv.most_similar(node, topn = 20000):
            n_all += 1

            
            if i[0] in Node_List:
                hit_list.append(i[0])
        performance_dict[(node,pair_node)] = hit_list 


    return performance_dict

def flatten(l):
    for x in l:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in flatten(x):
                yield y
        else:
            yield x


labels_in_hetio = ["Anatomy",
"BiologicalProcess",
"CellularComponent",
"Compound",
"Disease",
"Gene",
"MolecularFunction",
"Pathway",
"PharmacologicClass",
"SideEffect",
"Symptom"]
def compactWalks(pos_pairs, neg_pairs, s, t, k_nodes,k_val, kg="HetioNet"):
    #pair_1 = ['Canagliflozin', 'Dexamethasone']
    #pair_2 = ['Dapagliflozin','Betamethasone']
    #test_nodes = pair_1 + pair_2
    query_nodes = pos_pairs[0] + pos_pairs[1] + neg_pairs[0] + neg_pairs[1]
    query_nodes = list(set(query_nodes))
#    k_nodes = [["Gene","Disease"],["Gene"],[],[],[]]
    all_node_label = list(set(flatten([[s],[t],*k_nodes])))
#    all_node_label = labels_in_hetio
    cypher_query = buildCypher(s,t,k_nodes,k_val)
    print(cypher_query)
    #for each node in query_nodes build a subgraph based on the cypher query
    graph_uri = ""
    if(kg=="HetioNet"): graph_uri="bolt://neo4j.het.io"
    subgraph_dict = buildSubgraphDictonaryForNodes("bolt://neo4j.het.io", query_nodes, cypher_query, all_node_label, None,True)

    Walks = generateRandomWalks(subgraph_dict, query_nodes, 'deepwalk', 80, 5)
#    Walks = compactWalks(subgraph_dict, test_nodes, 'deepwalk', 80, 5)
    if(len(Walks)==0): return None, None
    model = buildModel(Walks) 
    eval_dic,pos_info_tuples  = evaluate(model, subgraph_dict, pos_pairs[0], pos_pairs[1], False)
    eval_dic,neg_info_tuples  = evaluate(model, subgraph_dict, neg_pairs[0], neg_pairs[1], False)
    return pos_info_tuples, neg_info_tuples

def compactWalks_v2(pos_pairs, neg_pairs, s, t, t_edges, k_nodes,k_edges,show_edges,k_val, kg="HetioNet",josh_mode=False):
    #pair_1 = ['Canagliflozin', 'Dexamethasone']
    #pair_2 = ['Dapagliflozin','Betamethasone']
    #test_nodes = pair_1 + pair_2
    query_nodes = pos_pairs[0] + pos_pairs[1] + neg_pairs[0] + neg_pairs[1]
    query_nodes = list(set(query_nodes))
#    k_nodes = [["Gene","Disease"],["Gene"],[],[],[]]
    all_node_label = list(set(flatten([[s],[t],*k_nodes])))
#    all_node_label = labels_in_hetio
    if(show_edges):
        cypher_query = buildCypherNodesAndEdges(s,t,t_edges,k_nodes,k_edges,k_val)
    else:
        cypher_query = buildCypher(s,t,k_nodes,k_val)

    print(cypher_query)
    #for each node in query_nodes build a subgraph based on the cypher query
    graph_uri = ""
    if(kg=="HetioNet"): graph_uri="bolt://neo4j.het.io"
    if(kg=="ROBOKOP"): graph_uri="bolt://robokopkg.renci.org:7687"
    subgraph_dict = buildSubgraphDictonaryForNodes(graph_uri, query_nodes, cypher_query, all_node_label, None,True)

    Walks = generateRandomWalks(subgraph_dict, query_nodes, 'deepwalk', 80, 5)
#    Walks = compactWalks(subgraph_dict, test_nodes, 'deepwalk', 80, 5)
    if(len(Walks)==0): return None, None
    model = buildModel(Walks) 
    if(josh_mode):
        pos_info_tuples  = evaluate_v2(model, subgraph_dict, pos_pairs[0], pos_pairs[1], False)
        neg_info_tuples  = evaluate_v2(model, subgraph_dict, neg_pairs[0], neg_pairs[1], False)
        return pos_info_tuples, neg_info_tuples
    eval_dic,pos_info_tuples  = evaluate(model, subgraph_dict, pos_pairs[0], pos_pairs[1], False)
    eval_dic,neg_info_tuples  = evaluate(model, subgraph_dict, neg_pairs[0], neg_pairs[1], False)
    return pos_info_tuples, neg_info_tuples

if(__name__=="__main__"):
    pair_1 = ['Canagliflozin', 'Dexamethasone']
    pair_2 = ['Dapagliflozin','Betamethasone']
    x = compactWalks([pair_1,pair_2],[[],[]],"Compound",None,[[]],1)
    print(x)
  
