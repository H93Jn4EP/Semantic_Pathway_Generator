from dash import Dash
import time
import dash_daq as daq
from itertools import combinations, product
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import base64
import io
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from compact_walks import compactWalks_v2

rk_nodes = []
with open("rk_nodes.txt") as f:
    for line in f: rk_nodes.append(line.strip())



hetio_nodes = ["Anatomy",
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


kg = "HetioNet"
kg = "ROBOKOP"


if(kg=="HetioNet"):  rk_nodes = hetio_nodes

rk_edges = []
with open("rk_edges.txt") as f:
    for line in f: rk_edges.append(line.strip())

hetio_edges = []
with open("hetio_edges.txt") as f:
    for line in f: hetio_edges.append(line.strip())

if(kg=="HetioNet"):  rk_edges = hetio_edges

app = Dash(__name__)

kg_drop = dcc.Dropdown(
    id="kg-dropdown",
   options=[
       {'label':"ROBOKOP", 'value':"ROBOKOP"},
       {'label':"HetioNet", 'value':"HetioNet"} 
   ],
   value="ROBOKOP",
   clearable=False
)

start_drop = dcc.Dropdown(
    id="start-dropdown",
   options=[
       {'label':x, 'value':x} for x in rk_nodes 
   ],
   #value="chemical_substance",
   value="Compound",
   clearable=False
)

tail_drop = dcc.Dropdown(
    id="tail-dropdown",
   options=[
       {'label':x, 'value':x} for x in rk_nodes 
   ]#,
   #value="disease"
)

node_drop = dcc.Dropdown(
    id="node-dropdown",
   options=[
       {'label':x, 'value':x} for x in rk_nodes 
       ],
   multi=True
)

#Make the 5 divs containing the node drop down. These also contain
# a bold header saying "Level k-1:". These need to be in seperate
# divs so they can easily be hidden and unhidden when k-val is changed.
k_drop = []
for k in range(1,6):
    edge_drop = html.Div([
        dcc.Dropdown(
        id="edge-dropdown-%i" % k,
            options=[
                {'label':x, 'value':x} for x in rk_edges 
            ],
        multi=True
        )
        
        ],
        id="edge-div-%i"%k,
        style={'display':'block'}
    )
    drop = html.Div([
        html.B(children='Level k-%i:'%k),
        edge_drop,
        dcc.Dropdown(
        id="node-dropdown-%i" % k,
            options=[
                {'label':x, 'value':x} for x in rk_nodes 
            ],
        multi=True
        )
        
    ],
    id="node-div-%i"%k,
    style={'display':'block'}
    )
    k_drop.append(drop)

#Make the 5 divs containing the edge drop down. These need
# to be in divs so they can easily be hidden and unhidden when k-val 
# is changed.
'''
k_edge_drop = []
for k in range(1,6):
    edge_drop = html.Div([
        dcc.Dropdown(
        id="edge-dropdown-%i" % k,
            options=[
                {'label':x, 'value':x} for x in rk_edges 
            ],
        multi=True
        )
        
    ],
    id="edge-div-%i"%k,
    style={'display':'block'}
    )
    k_edge_drop.append(edge_drop)
    '''

edge_drop = dcc.Dropdown(
    id="edge-dropdown",
   options=[
       {'label':x, 'value':x} for x in rk_edges 
   ],
   multi=True,
   style={'display':'None'}
)

tail_edge = dcc.Dropdown(
    id="tail-edge",
   options=[
       {'label':x, 'value':x} for x in rk_edges 
   ],
   multi=True
)

k_sel = daq.NumericInput(
   id="k-select",
   min=1,
   max=5,
   value=2
) 

pos_pairs = html.Div([
    html.Div(html.B(children='Positive Pairs:')),
    dcc.Textarea(
        id='positive_pairs',
        value='''Canagliflozin,Dapagliflozin
Dexamethasone,Betamethasone
Lapatinib,Afatinib
Captopril,Enalapril
Losartan,Valsartan
Nifedipine,Felodipine
Simvastatin,Atorvastatin
Alendronate,Incadronate
Citalopram,Escitalopram''',
        style={'width': '20%', 'height': 140, 'width': 300},
)])

neg_pairs = html.Div([
    html.Div(html.B(children='Negative Pairs:\n')),
    dcc.Textarea(
        id='negative_pairs',
        value='''Dexamethasone,Canagliflozin
Afatinib,Captopril
Escitalopram,Losartan
Betamethasone,Enalapril
Dapagliflozin,Nifedipine
Citalopram,Felodipine''',
        style={'width': '20%', 'height': 140, 'width': 300},
)])


load =  dcc.Loading(
   id="loading-1",
   type="default",
   children=html.Div(id="compactwalk-output")
)

row1 = html.Tr([html.Td(html.Div([html.Div(id='dd-output-container'),
    html.Img(id='img-test'),
    pos_pairs, 
    neg_pairs,
    html.Button('Submit', id='submit-val', n_clicks=0)])),
    html.Td(load)])

edge_checkbox = dcc.Checklist(id="edge-checkbox",options=[{"label":"Use Edges","value":"True"}],value=[])

node_edge_div = html.Div([
    #k_edge_drop[0],
    k_drop[0],
    #k_edge_drop[1],
    k_drop[1],
    #k_edge_drop[2],
    k_drop[2],
    #k_edge_drop[3],
    k_drop[3],
    #k_edge_drop[4],
    k_drop[4]
    ],
    id="node-edge-div")

selector = html.Div([
    html.B(children='Start Node:'),
    start_drop,
    html.B(children='Tail Node:'),
    tail_edge,
    tail_drop,
    html.B(children='K Value:'),
    k_sel,
    edge_checkbox,
    node_edge_div,
#    html.H4(children='Pathway Node Labels:'),
#    node_drop,
#    html.H4(children='Pathway Edge Labels:'),
#    edge_drop,
#    html.Div(id='edge-output-container'),
#    html.Div(dcc.Input(id='input-on-submit', type='text')),
#    html.Div(id='container-button-basic', children='Enter a value and press submit')
#    table
     html.Div(id="test-output")
])

row0 = html.Tr([html.Td(selector)])
tbody = html.Tbody([row0, row1])
table = html.Table(tbody)

app.layout = html.Div([table])


selected_nodes = []



selected_edges = []



def buildCypherJustNodes(s,t,k_nodes,k_val):
    query = "MATCH "
    query += "(s:`%s`)" % s
    for k in range(k_val):
        query += "--(x%i)" % (k+1)
    if(t!=None): query+= "--(t:`%s`)" % t

    label_clauses = []
    for k in range(k_val):
        if(k_nodes[k]==None or len(k_nodes[k])==0):continue
        #where_str += ( ) 
        clauses = ["x%i:`%s`" % (k+1,label) for label in k_nodes[k]]
        clause_str = " ( " + " OR ".join(clauses) + " ) "
        label_clauses.append(clause_str)
    if(len(label_clauses)!=0):
        query += " WHERE "
        query += " AND ".join(label_clauses)
    query += " RETURN * " 
    return query

def buildCypherNodesAndEdges(s,t,t_edges,k_nodes,k_edges,k_val):
    query = "MATCH "
    query += "(s:`%s`)" % s
    for k in range(k_val):
        query += "-[r%i]-(x%i)" % (k+1,k+1)
    if(t!=None): query+= "-[rt]-(t:`%s`)" % t

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
    #for k in range(k_val):
    #    if(k_edges[k]==None or len(k_edges[k])==0):continue
    #    #where_str += ( ) 
    #    clauses = ["r%i:`%s`" % (k+1,label) for label in k_edges[k]]
    #    clause_str = " ( " + " OR ".join(clauses) + " ) "
    #    label_clauses.append(clause_str)
    #    #where_str += ( ) 
    #if(t!=None and  len(t_edges)!=0):
    #    clauses = ["rt:`%s`" % (label) for label in t_edges]
    #    clause_str = " ( " + " OR ".join(clauses) + " ) "
    #    label_clauses.append(clause_str)
    if(len(label_clauses)!=0):
        query += " WHERE "
        query += " AND ".join(label_clauses)
    query += " RETURN * " 
    return query


def multiSelectHelper(l):
    if(l!=None and len(l)>0): return l
    else: return ["*"]

@app.callback(
        [Output('tail-edge','disabled'),
         Output('tail-edge','value')],
        [Input("tail-dropdown", 'value')],
)
def tailOuput(tail_val):
    if(tail_val==None): return True,[] 
    else: return False,[]

@app.callback(
    [
        Output('img-test', 'src'),
        Output('dd-output-container', 'children')
    ],
    [
        Input("start-dropdown", 'value'), 
        Input("tail-dropdown", 'value'), 
        Input('tail-edge','value'),
        Input("node-dropdown-1", 'value'), 
        Input("node-dropdown-2", 'value'), 
        Input("node-dropdown-3", 'value'), 
        Input("node-dropdown-4", 'value'), 
        Input("node-dropdown-5", 'value'),
        Input("edge-dropdown-1", 'value'), 
        Input("edge-dropdown-2", 'value'), 
        Input("edge-dropdown-3", 'value'), 
        Input("edge-dropdown-4", 'value'), 
        Input("edge-dropdown-5", 'value'),
        Input('edge-checkbox', 'value'),
        Input('k-select', 'value')
    ]
)
def cypher_output(s,t,t_edges,k1_nodes,k2_nodes,
        k3_nodes,k4_nodes,k5_nodes,
        k1_edges,k2_edges,
        k3_edges,k4_edges,k5_edges,show_edge,
        k_val):
    show_edge = checkToBool(show_edge)
    k_nodes = [k1_nodes,k2_nodes,k3_nodes,k4_nodes,k5_nodes]
    k_edges = [k1_edges,k2_edges,k3_edges,k4_edges,k5_edges]
    if(s=="chemical_substance" and k1_nodes==None and k2_nodes==None and k_val==2 and not show_edge):
        query = buildCypherJustNodes(s,t,[None,None,None,None,None],k_val)
        with open("base_image.txt") as f:
            image = f.read()
        return (image,query)

       # return "data:image/png;base64,{}".format(''), ""
    node_layers = []
    if(k_val >= 1 and show_edge): node_layers.append(multiSelectHelper(k1_edges))
    if(k_val >= 1): node_layers.append(multiSelectHelper(k1_nodes))
    if(k_val >= 2 and show_edge): node_layers.append(multiSelectHelper(k2_edges))
    if(k_val >= 2): node_layers.append(multiSelectHelper(k2_nodes))
    if(k_val >= 3 and show_edge): node_layers.append(multiSelectHelper(k3_edges))
    if(k_val >= 3): node_layers.append(multiSelectHelper(k3_nodes))
    if(k_val >= 4 and show_edge): node_layers.append(multiSelectHelper(k4_edges))
    if(k_val >= 4): node_layers.append(multiSelectHelper(k4_nodes))
    if(k_val >= 5 and show_edge): node_layers.append(multiSelectHelper(k5_edges))
    if(k_val >= 5): node_layers.append(multiSelectHelper(k5_nodes))
    if(show_edge and t!=None): node_layers.append(multiSelectHelper(t_edges))
    fig, ax = plt.subplots(1,1)
    buf = io.BytesIO() # in-memory files
    pairs_list = []
    print("NODE LAYERS:" , node_layers)
    if(k_val==1 and not show_edge):
        pairs_list.append([(n+"1",None) for n in node_layers[0]])
    else:
        for i in range(len(node_layers)-1):
            pairs = list(product(node_layers[i],node_layers[i+1]))
            if(show_edge):
                depth = i//2 + 1
                #If i is even, were on a node -> rels layer
                if(i%2==0):
                    pairs = [(x+"-r"+str(depth),y+"-n"+str(depth)) for (x,y) in pairs]
                elif(i%2==1 and i==len(node_layers)-2):
                    pairs = [(x+"-n"+str(depth),y+"-r-t") for (x,y) in pairs]
                else:
                    pairs = [(x+"-n"+str(depth),y+"-r"+str(depth+1)) for (x,y) in pairs]

            else:
                pairs = [(x+str(i+1),y+str(i+2)) for (x,y) in pairs]
            pairs_list.append(pairs)
            #pairs_list.append(list(product(node_layers[i],node_layers[i+1])))
    COP_Visualize_v2(ax, s, t, pairs_list)
        
    fig.savefig(buf, format = "png") # save to the above file object
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    image = "data:image/png;base64,{}".format(data)
    if(show_edge):
        query = buildCypherNodesAndEdges(s,t,t_edges,k_nodes,k_edges,k_val)
    else:
        query = buildCypherJustNodes(s,t,k_nodes,k_val)
    return (image, query)

def COP_Visualize_v2(ax,start,tail,pairs_list):
    nodes = set()
    nodes.add(start)
    if(tail!=None):
        tail = tail + '-t'
        nodes.add(tail)
    for count, pairs in enumerate(pairs_list):
        for (x,y) in pairs:
            nodes.add(x)
            if(y!=None): nodes.add(y)

    nodes = list(nodes)

    colors = []
    pos_dict={start:(0,0)}
    for n in nodes:
        if(n.endswith("1")): 
            colors.append("lightblue")
        elif(n.endswith("2")): 
            colors.append("pink")
        elif(n.endswith("3")): 
            colors.append("lightgreen")
        elif(n.endswith("4")): 
            colors.append("lightcoral")
        elif(n.endswith("5")): 
            colors.append("plum")
        else: colors.append("tan")
    G=nx.DiGraph()
    G.add_nodes_from(nodes)
    print(G.nodes())
    
    for count, pairs in enumerate(pairs_list):
        depth = count+1 #Want a depth value that shows how far we are from start. Each pair in pairs_list is the next layer of depth (with seed as 0th layer).

       # print('len',len(pairs_list))
       # print('dep',depth)
       # print('tail',tail)
       
        #If were on the first row of pairs and theres only one row, link it 
        # to the source and tail. 
        if(count==0 and len(pairs_list)==1):
            for (x,y) in pairs:
                G.add_edge(start,x)
                if(y==None and tail!=None): G.add_edge(x,tail)
                if(y!=None): G.add_edge(x,y)
                if(y!=None and tail!=None): G.add_edge(y,tail)
        # If were on the first row, link it to the source.
        elif(count==0):
            for (x,y) in pairs:
                G.add_edge(start,x)
                if(y!=None): G.add_edge(x,y)
        # If were on the last row and there is a tail, link the last row to the tail.
        elif(count==len(pairs_list)-1 and tail!=None):
            print("HI")
            for (x,y) in pairs:
                G.add_edge(x,y)
                G.add_edge(y,tail)
        #If were on any other row, add it as edges.
        else:
            for (x,y) in pairs:
                G.add_edge(x,y)

    print("EDGES:")
    print(G.edges())
    pos = graphviz_layout(G, prog="dot",root=start)
    print("POS:")
    print(pos)
#    print(nx.is_tree(G))
#    print(colors)
#    nx.draw(G, with_labels = True, pos=pos, ax=ax, node_color=colors, node_size=1000)
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=colors, node_size=1000)
    nx.draw_networkx_labels(G, pos=pos)
    nx.draw_networkx_edges(G, pos=pos)
#    nx.draw_networkx_edge_labels(G, pos=pos,edge_labels={("Seed","Gene1"):"HI"})
    ax.set_xlim(ax.get_xlim()[0]-15,ax.get_xlim()[1]+15)

def COP_Visualize(ax,start,tail,pairs_list,show_edge):
    nodes = set()
    nodes.add(start)
    if(tail!=None):
        nodes.add(tail)
    for count, pairs in enumerate(pairs_list):
        depth_x = str(count+1) #Want a depth value that shows how far we are from start. Each pair in pairs_list is the next layer of depth (with seed as 0th layer).
        depth_y = str(depth_x+1) 
        for (x,y) in pairs:
            nodes.add(x+depth_x)
            if(y!=None): nodes.add(y+depth_y)

    nodes = list(nodes)

    colors = []
    pos_dict={start:(0,0)}
    for n in nodes:
        if(n.endswith("1")): 
            colors.append("lightblue")
        elif(n.endswith("2")): 
            colors.append("pink")
        elif(n.endswith("3")): 
            colors.append("lightgreen")
        elif(n.endswith("4")): 
            colors.append("lightcoral")
        elif(n.endswith("5")): 
            colors.append("plum")
        else: colors.append("tan")
    G=nx.DiGraph()
    G.add_nodes_from(nodes)
    print(G.nodes())
    
    for count, pairs in enumerate(pairs_list):
        depth = count+1 #Want a depth value that shows how far we are from start. Each pair in pairs_list is the next layer of depth (with seed as 0th layer).
        print('len',len(pairs_list))
        print('dep',depth)
        print('tail',tail)
        if(depth==1 and len(pairs_list)==1):
            for (x,y) in pairs:
                G.add_edge(start,x+str(depth))
                if(y==None and tail!=None): G.add_edge(x+str(depth),tail)
                if(y!=None): G.add_edge(x+str(depth),y+str(depth+1))
                if(y!=None and tail!=None): G.add_edge(y+str(depth+1),tail)
        elif(depth==1):
            for (x,y) in pairs:
                G.add_edge(start,x+str(depth))
                if(y!=None): G.add_edge(x+str(depth),y+str(depth+1))
        elif(depth==len(pairs_list) and tail!=None):
            print("HI")
            for (x,y) in pairs:
                G.add_edge(x+str(depth),y+str(depth+1))
                G.add_edge(y+str(depth+1),tail)
        else:
            for (x,y) in pairs:
                G.add_edge(x+str(depth),y+str(depth+1))

    pos = graphviz_layout(G, prog="dot",root=start)
    print(nx.is_tree(G))
    print(colors)
#    nx.draw(G, with_labels = True, pos=pos, ax=ax, node_color=colors, node_size=1000)
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=colors, node_size=1000)
    nx.draw_networkx_labels(G, pos=pos)
    nx.draw_networkx_edges(G, pos=pos)
#    nx.draw_networkx_edge_labels(G, pos=pos,edge_labels={("Seed","Gene1"):"HI"})
    ax.set_xlim(ax.get_xlim()[0]-15,ax.get_xlim()[1]+15)

def checkToBool(show_edge):
    if(len(show_edge)==1): return True
    else: return False

@app.callback(
    [
    Output("node-div-1",'style'),
    Output("node-div-2",'style'),
    Output("node-div-3",'style'),
    Output("node-div-4",'style'),
    Output("node-div-5",'style'),
    Output("edge-div-1",'style'),
    Output("edge-div-2",'style'),
    Output("edge-div-3",'style'),
    Output("edge-div-4",'style'),
    Output("edge-div-5",'style'),
    Output("tail-edge",'style')
    ],
    [Input('k-select', 'value'),
    Input('edge-checkbox', 'value')]
)
def hide_elements(k,show_edge):
    show_edge = checkToBool(show_edge)
    style_1 = {'display':'block'} if k>=1 else {'display':'None'}
    style_2 = {'display':'block'} if k>=2 else {'display':'None'}
    style_3 = {'display':'block'} if k>=3 else {'display':'None'}
    style_4 = {'display':'block'} if k>=4 else {'display':'None'}
    style_5 = {'display':'block'} if k>=5 else {'display':'None'}

    edge_style_1 = {'display':'block'} if show_edge and k>=1 else {'display':'None'}
    edge_style_2 = {'display':'block'} if show_edge and k>=2 else {'display':'None'}
    edge_style_3 = {'display':'block'} if show_edge and k>=3 else {'display':'None'}
    edge_style_4 = {'display':'block'} if show_edge and k>=4 else {'display':'None'}
    edge_style_5 = {'display':'block'} if show_edge and k>=5 else {'display':'None'}
    
    tail_edge_style = {'display':'block'} if show_edge else {'display':'None'}

    return style_1, style_2, style_3, style_4, style_5, edge_style_1, edge_style_2, edge_style_3, edge_style_4, edge_style_5, tail_edge_style

import csv
def processPairText(text):
    l1 = []
    l2 = []
    for line in text.split('\n'):
        a = line.split(',')
        if(len(a)<2):continue
        l1.append(a[0].strip())
        l2.append(a[1].strip())
    return l1,l2


#@app.callback(
#    Output('test-output', 'children'),
#    Input('node-edge-div', 'value')
#)
#def test(val):
#    print("Value of val",val)
#    return val 


@app.callback(
    Output('compactwalk-output', 'children'),
    Input('submit-val', 'n_clicks'),
    [
        State('positive_pairs', 'value'),
        State('negative_pairs','value'),
        State("start-dropdown", 'value'), 
        State("tail-dropdown", 'value'), 
        State('tail-edge','value'),
        State("node-dropdown-1", 'value'), 
        State("node-dropdown-2", 'value'), 
        State("node-dropdown-3", 'value'), 
        State("node-dropdown-4", 'value'), 
        State("node-dropdown-5", 'value'),
        State("edge-dropdown-1", 'value'), 
        State("edge-dropdown-2", 'value'), 
        State("edge-dropdown-3", 'value'), 
        State("edge-dropdown-4", 'value'), 
        State("edge-dropdown-5", 'value'),
        State('edge-checkbox', 'value'),
        State('k-select', 'value')
    ]
)
def submit_compact_walks(n_clicks,pos_pair_text,
        neg_pair_text,s,t,t_edges,k1_nodes,k2_nodes,
        k3_nodes,k4_nodes,k5_nodes,
        k1_edges,k2_edges,
        k3_edges,k4_edges,k5_edges,
        show_edges,
        k_val):#, sel_nodes, sel_edges):
    if(n_clicks <= 0): return "" 
    print("Running COMPACT WALKS!")
    show_edges = checkToBool(show_edges)
    pos_pairs = processPairText(pos_pair_text)
    neg_pairs = processPairText(neg_pair_text)
    k_nodes = [k1_nodes,k2_nodes,k3_nodes,k4_nodes,k5_nodes]
    k_edges = [k1_edges,k2_edges,k3_edges,k4_edges,k5_edges]
#    pos_info_tuples, neg_info_tuples = compactWalks(pos_pairs,neg_pairs,s,t,k_nodes,k_val)
    pos_info_tuples, neg_info_tuples = compactWalks_v2(pos_pairs, neg_pairs, s, t, t_edges, k_nodes,k_edges,show_edges,k_val,kg)
    if(pos_info_tuples==None): return html.B("Pathway could not be matched in the graph.")
#def compactWalks(pos_pairs, neg_pairs, s, t, k_nodes,k_val):
    l = [html.B("POSITIVE PAIRS:"),html.Br()]
    mrrs = []
    for (n1,n2,rank, tot) in pos_info_tuples:
        try:
           mrr_val = (1/int(rank))
           mrrs.append(mrr_val)
        except ValueError:
            continue
    print('sum mrr:', sum(mrrs))
    l.append("MRR: %f" % ((sum(mrrs) * 1.0)/len(mrrs)))
    l.append(html.Br())
    for (n1,n2,rank, tot) in pos_info_tuples:
        info_str = "Rank of %s compared to  %s: %i out of %i" %(n1,n2,rank, tot)
        l.append(info_str)
        l.append(html.Br())
    l.append(html.B("NEGATIVE PAIRS:"))
    l.append(html.Br())
    mrrs = []
    for (n1,n2,rank, tot) in neg_info_tuples:
        try:
           mrr_val = (1/int(rank))
           mrrs.append(mrr_val)
        except ValueError:
            continue
    print("MRRs",mrrs)
    l.append("MRR: %f" % ((sum(mrrs) *1.0)/len(mrrs)))
    l.append(html.Br())
    for (n1,n2,rank, tot) in neg_info_tuples:
        info_str = "Rank of %s compared to  %s: %i out of %i" %(n1,n2,rank, tot)
        l.append(info_str)
        l.append(html.Br())


    return html.P(l[:-1])

#    return 'The input value was "{}" "{}" and the button has been clicked {} times'.format(
#        x,
#        y,
#        n_clicks
#    )

if __name__ == '__main__':
    app.run_server(host="0.0.0.0",port=80,debug=True)
