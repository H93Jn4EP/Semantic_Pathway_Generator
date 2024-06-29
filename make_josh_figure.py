from compact_walks import compactWalks_v2

hetio_edges = []
with open("hetio_edges.txt") as f:
    for line in f: hetio_edges.append(line.strip())
hetio_edges.append(None)

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
"Symptom",
None]

def processPairText(text):
    l1 = []
    l2 = []
    for line in text.split('\n'):
        a = line.split(',')
        if(len(a)<2):continue
        l1.append(a[0].strip())
        l2.append(a[1].strip())
    return l1,l2
pos_pair_text='''Canagliflozin,Dapagliflozin
Dexamethasone,Betamethasone
Lapatinib,Afatinib
Captopril,Enalapril
Losartan,Valsartan
Nifedipine,Felodipine
Simvastatin,Atorvastatin
Alendronate,Incadronate
Citalopram,Escitalopram'''

neg_pair_text='''Dexamethasone,Canagliflozin
Afatinib,Captopril
Escitalopram,Losartan
Betamethasone,Enalapril
Dapagliflozin,Nifedipine
Citalopram,Felodipine'''

show_edges = True

pos_pairs = processPairText(pos_pair_text)
neg_pairs = processPairText(neg_pair_text)
k_val=2
k1_nodes = ["SideEffect"]
k2_nodes = []
k3_nodes = []
k4_nodes = []
k5_nodes = []

k1_edges = []
k2_edges = []
k3_edges = []
k4_edges = []
k5_edges = []
#with open("josh_info.csv",'w',newline='') as csvfile:
#    x=1

def run_for_nodes(k1_nodes=[],k1_edges=[],k2_nodes=[],k2_edges=[],k3_nodes=[],k3_edges=[],k_val=2):
    k_nodes = [k1_nodes,k2_nodes,k3_nodes,k4_nodes,k5_nodes]
    k_edges = [k1_edges,k2_edges,k3_edges,k4_edges,k5_edges]
    # pos_info_tuples, neg_info_tuples = compactWalks(pos_pairs,neg_pairs,s,t,k_nodes,k_val)
    s = "Compound"
    t = None
    
    pos_info_tuples, neg_info_tuples = compactWalks_v2(pos_pairs, neg_pairs, s, t, k_nodes,k_edges,show_edges,k_val,"HetioNet",True)
    if(pos_info_tuples==None):pos_info_tuples={}
    if(neg_info_tuples==None):neg_info_tuples={}
    import csv
    with open("josh_info_2_hop.csv",'a',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        mrr = 0.0
        mrr_cnt = 0
        for (n1,n2) in pos_info_tuples    :
            l = pos_info_tuples[(n1,n2)]
            try:
                rank = l.index(n2) + 1
                mrr += (1.0/rank)
                mrr_cnt += 1
            except ValueError:
                x=1
        if(mrr_cnt==0): mrr=0
        else: mrr = mrr / mrr_cnt
        csvwriter.writerow(["POS",mrr,"COMPOUND",*k1_nodes,*k1_edges,*k2_nodes,*k2_edges,*k3_nodes,*k3_edges])
    
        for (n1,n2) in pos_info_tuples:
            l = pos_info_tuples[(n1,n2)]
            try:
                rank = l.index(n2) + 1
            except:rank=0
            csvwriter.writerow([n1,n2,rank,*l])

        mrr = 0.0
        mrr_cnt = 0
        for (n1,n2) in neg_info_tuples:
            l = neg_info_tuples[(n1,n2)]
            try:
                rank = l.index(n2) + 1
                mrr += (1.0/rank)
                mrr_cnt += 1
            except ValueError:
                x=1
        if(mrr_cnt==0): mrr=0
        else: mrr = mrr / mrr_cnt
        csvwriter.writerow(["NEG",mrr,"COMPOUND",*k1_nodes,*k1_edges,*k2_nodes,*k2_edges,*k3_nodes,*k3_edges])

        for (n1,n2) in neg_info_tuples:
            l = neg_info_tuples[(n1,n2)]
            try:
                rank = l.index(n2) + 1
            except:rank=0
            csvwriter.writerow([n1,n2,rank,*l])
 
for n1 in labels_in_hetio:
    for r1 in hetio_edges:
        for n2 in labels_in_hetio:
            for r2 in hetio_edges:
                if(n1==None):k1_nodes=[]
                else:k1_nodes = [n1]
    
                if(r1==None):k1_edges=[]
                else:k1_edges=[r1]
                
                if(n2==None):k2_nodes=[]
                else:k2_nodes = [n2]
    
                if(r2==None):k2_edges=[]
                else:k2_edges=[r2]
                run_for_nodes(k1_nodes,k1_edges,k2_nodes,k2_edges,k3_nodes,k3_edges,k_val=2)
