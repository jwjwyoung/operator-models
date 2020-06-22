import sys
import traceback
sys.path.append('/home/azureuser/.ipython/extensions/')
from collections import *
from functools import cmp_to_key
import pandas as pd
import numpy as np
from sklearn import tree,linear_model
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
import pickle
import signal
import itertools
import re
from column_summary import * 
import time
import random

_tables = []
for fname in os.listdir('./'):
    if not fname.endswith('.pkl'): 
        continue
    if fname.startswith('col_summary_withsampl'):
        print(fname)
        with open(fname, 'rb') as fr:
            try:
                while True:
                    _tables.append(pickle.load(fr))
            except EOFError:
                pass
        #break

# find out if ground truth columns are indeed contiguous in the table
def is_melt_cols_contiguous(vec):
    is_cols_contiguous_counter = Counter()
    for t in vec:
        melt_cols = []
        for c in t:
            if(c.is_value):
                melt_cols.append(c.col_pos)

        #print(melt_cols)
        if(len(melt_cols) == 0):
            continue
        is_contiguous = (max(melt_cols) - min(melt_cols) + 1) == len(melt_cols)
        #print(is_contiguous)
        is_cols_contiguous_counter.update({is_contiguous:1})

    print('is_cols_contiguous_counter: {}'.format(is_cols_contiguous_counter.update))

# find out if ground truth columns are indeed contiguous in the table 
is_melt_cols_contiguous(_tables)
#exit(0)        
        
        
def shuffle_and_split(vec):
    random_seed = 0
    random.Random(random_seed).shuffle(vec)
    datamap = {}
    fail0 = 0
    fail1 = 0
    fail2 = 0
    fail3 = 0
    notebook_data_count = Counter()
    reject_notebook = []
    for t in vec:
        if len(t) == 0:
            continue
        filename = t[0].filename
        idx = re.search(r"_cell[0-9]*.pickle", filename).start()
        notebook = str(filename)[:idx]
        notebook_data_count.update([notebook])
    for k,v in notebook_data_count.most_common():
        if v > 300:
            reject_notebook.append(k)
            print("Reject {}: {}".format(k,v))
    for t in vec:
        if len(t)==0:
            fail2 += 1
            continue
        filename = t[0].filename
        idx = re.search(r"_cell[0-9]*.pickle", filename).start()
        notebook = str(filename)[:idx]
        if notebook in reject_notebook:
            fail0 += 1
            continue
        if not any([c.is_id or c.is_value for c in t]):
            fail1 += 1
            continue
        if t[0].sz < 5: #not (t[0].sz > 5 and any([c.is_head for c in t])):
            fail3 += 1
            continue
        #key = '{}---{}'.format(t[0].filename, t[0].data_index)
        datamap_key = '{}:{}'.format('--'.join([str(c.col_name) for c in t]), t[0].sz)
        if datamap_key not in datamap:
            datamap[datamap_key] = []
        datamap[datamap_key].append(t)
    
    print("fail0 = {}, fail1 = {}, fail2 = {}, fail3 = {}".format(fail0, fail1, fail2, fail3))
    merged_raw = []
    sum_filtered = 0 
    param_id_counter = Counter()
    Nnotebooks = set()
    filename_lookup = []
    _shape_vec = Counter()
    for k,v in datamap.items():
        t = v[0]
        sum_filtered += len(v)
        param_id_set = set()
        for table in v:
            filename = table[0].filename
            idx = re.search(r"_cell[0-9]*.pickle", filename).start()
            repo = str(filename)[:idx]
            repo = repo.replace('_datadrive','').replace('_mnt','')
            Nnotebooks.add(repo)
            id_sig = '-'.join(filter(lambda x: x is not None, ['{}({}{})'.format(c.col_name, 'i' if c.is_id else '', 'v' if c.is_value else '') if (c.is_id or c.is_value) else None for c in table]))
            #print(id_sig)
            param_id_set.add(id_sig)
        if len(v) > 0:
            param_id_counter.update({len(param_id_set):1})
        for i in range(len(t)): 
            t[i].is_id = any([v[x][i].is_id for x in range(len(v))])
            t[i].is_value = any([v[x][i].is_value for x in range(len(v))])
        #if len(v) > 1:
        #    print("sum index = {}; sum_head = {} / len_tables = {}".format(sum([t[i].is_index for i in range(len(t))]),sum([t[i].is_head for i in range(len(t))]), len(v)))
        for c in t:
            if c.shapes is not None:
                _shape_vec.update(c.shapes)
        merged_raw.append(t)
        filename_lookup.append([(tt[0].filename, tt[0].data_index) for tt in v])
    
    print("orig table = {}, after filter = {}, after merged = {}".format(len(vec), sum_filtered, len(merged_raw)))
    print("Nnotebooks = {}".format(len(Nnotebooks)))
    print("id counter = {}".format(param_id_counter))
    
    return _shape_vec,merged_raw,filename_lookup

_shape_vec,tables,_filename_lookup = shuffle_and_split(_tables)

print("avg columns per table = {}".format(float(sum([len(t) for t in tables])) / float(len(tables))))
print("avg id per table = {}".format(float(sum([sum([int(c.is_id) for c in t]) for t in tables])) / float(len(tables))))
print("avg value per table = {}".format(float(sum([sum([int(c.is_value) for c in t]) for t in tables])) / float(len(tables))))
shape_vec_helper = _shape_vec.most_common()[:5]
#print(shape_vec_helper)
shape_vec = [v[0] for v in shape_vec_helper]

splitv = int(len(tables)*0.8)
print("split: {}".format(splitv))

train = tables[:splitv]
test = tables[splitv:]
filename_lookup = _filename_lookup[splitv:]
assert(len(test)==len(filename_lookup))


#train = train[:40]
#test = test[:10]
# model for column compatibility
# linear model
train_X = []
train_y = []
pos_cnt = Counter()
start = time.time()
weight = []
for ii,table in enumerate(train):
    f,y,pairs = get_features_for_compatibility(table, True)
    if len(y) == 0:
        continue
    if sum(y) == 0:
        neg_weight = 1000.0 / float(len(y))
        pos_weight = 1.0
    else:
        neg_weight = (1000.0 / float(len(y) - sum(y))) if len(y) - sum(y) > 0 else 1.0
        pos_weight = 1000.0 / float(sum(y))
    #print("{} - {}: negweight = {}, posweight = {}".format(len(y), sum(y), neg_weight, pos_weight))
    weight += [pos_weight if y[i] else neg_weight for i in range(len(y))]
    pos_cnt.update(y)
    if ii % 20 == 0:
        print("table {}; feature size = {} / pos = {} / time = {}".format(ii, len(train_X), pos_cnt, time.time()-start))
    train_X += f
    train_y += y
    
#model = LinearRegression()
model = XGBRegressor(
        max_depth=4,
        #objective='reg:linear',
        #objective='reg:logistic',
        seed=0
        )

model.fit(train_X, train_y, sample_weight=weight)
pickle.dump(model, open("model.pkl",'wb'))

if(isinstance(model, LinearRegression)):
    print("coef = {}".format(model.coef_))
#exit(0)

print("finish training")
logf = open("train_log.log", 'w')
def write_log(s):
    logf.write(s+'\n')
    print(s)
def flush_log():
    logf.write("\n")
    logf.flush()
class Edge(object):
    def __init__(self, edge, node1, node2, eid):
        self.edge = edge
        self.node1 = node1
        self.node2 = node2
        self.eid = eid
class IterState(object):
    def __init__(self, min_ingroup, max_connect, new_node):
        self.min_ingroup = min_ingroup
        self.max_connect = max_connect  
        self.score = min_ingroup - max_connect
        self.new_node = new_node

def get_min_edge(l):
    return min([e.edge for e in l])
def get_avg_edge(l):
    if len(l) == 0:
        return 1 # return 1 for now, eval last cluster by significance of previous scores (if no previous iteration/partition has in-out score > 0.55, then merge all into last cluster)
    return float(sum([e.edge for e in l])) / float(len(l))
def get_max_edge(l, withnode=False):
    if withnode == False:
        return max([e.edge for e in l])
    else:
        pair = (l[0].edge, l[0])
        for e in l[1:]:
            if e.edge > pair[0]:
                pair = (e.edge,e)
        return pair
TERMINATE_THRESHOLD = 0.4
INCLUDELAST_THRESHOLD = 0.3
def get_the_value_group(table, model):
    if len(table) > 500:
        print("large table removed {}".format(len(table)))
        return []
    features,_y,pairs = get_features_for_compatibility(table)
    if len(pairs) == 1:
        type_ids = [col.dtype_to_int(col.dtype) for col in table]
        _y_predict = [type_ids[i]==min(type_ids) for i in range(len(type_ids))]
        return _y_predict
    if len(features) == 0:
        return []
    y = model.predict(features)
    sorted_y = [(ii,f) for ii,f in enumerate(y)]
    sorted_y.sort(key = lambda x: 0.0-x[1])
    edges = []
    ideal_ingroup = []
    ideal_value_group = set()
    connects = []
    for ii,f in sorted_y:
        e = Edge(f, pairs[ii][0], pairs[ii][1], len(edges))
        edges.append(e)
        if pairs[ii][0].is_value and pairs[ii][1].is_value:
            ideal_ingroup.append(e)
            ideal_value_group.add(pairs[ii][0])
            ideal_value_group.add(pairs[ii][1])
        elif int(pairs[ii][0].is_value) + int(pairs[ii][1].is_value) == 1:
            connects.append(e)
    if len(ideal_ingroup) == 0:
        write_log("No node in group")
        _y_predict = [col.is_value for col in table]
        return _y_predict
    else:
        avg_ideal_ingroup = get_avg_edge(ideal_ingroup)
        avg_ideal_outgroup = get_avg_edge(connects)
        max_ingroup, maxedge = get_max_edge(ideal_ingroup, True) 
        write_log("ideal starts = {:.4f} ({} - {}); group_sz = (in: {}/connect: {}/value: {}); score = {:.4f} - {:.4f} = {:.4f}".format(max_ingroup, maxedge.node1.col_name, maxedge.node2.col_name, len(ideal_ingroup), len(connects), len(ideal_value_group), avg_ideal_ingroup, avg_ideal_outgroup, avg_ideal_ingroup-avg_ideal_outgroup)) 
    #value_group = set(list(pairs[sorted_y[0][0]]))
    #in_group = set([edges[0]])
    #out_group = set(edges[1:])

    # sample starting point from user
    random.shuffle(edges)
    starte = None
    for e in edges: 
        if e.node1.is_value and e.node2.is_value:
            starte = e
    assert(starte is not None)
    edges.remove(starte)
    value_group = set([starte.node1, starte.node2])
    in_group = set([starte])
    out_group = set(edges)
    avg_ingroup = get_avg_edge(in_group)
    connects = []
    for e in out_group:
        x1 = e.node1 in value_group
        x2 = e.node2 in value_group
        if int(x1)+int(x2) == 1:
            connects.append(e)
    avg_outgroup = get_avg_edge(connects)
    iterstates = [IterState(avg_ingroup, avg_outgroup, None)]
    assert(len(value_group)==2)
    #write_log("start group = {} {} / score = {:.4f} - {:.4f} = {:.4f}".format(pairs[sorted_y[0][0]][0].col_name, pairs[sorted_y[0][0]][1].col_name, avg_ingroup, avg_outgroup, avg_ingroup-avg_outgroup))
    write_log("start group = {} {} / score = {:.4f} - {:.4f} = {:.4f}".format(starte.node1.col_name, starte.node2.col_name, avg_ingroup, avg_outgroup, avg_ingroup-avg_outgroup))
    last_round = avg_ingroup
    while len(out_group) > 0:
        # find all the connecting edges:
        connects = []
        for e in out_group:
            x1 = e.node1 in value_group
            x2 = e.node2 in value_group
            if int(x1)+int(x2) == 1:
                connects.append(e)
        if len(connects) == 0:
            break
        max_connect,max_connect_edge = get_max_edge(connects, True)
        col1 = max_connect_edge.node1
        col2 = max_connect_edge.node2
        new_node = col1 if col1 not in value_group else col2
        old_node = col2 if col1 not in value_group else col1
        value_group.add(col1)
        value_group.add(col2)
        for e in [e1 for e1 in out_group]:
            if e.node1 in value_group and e.node2 in value_group:
                out_group.remove(e)
                in_group.add(e)
        #connects = []
        #for e in out_group:
        #    x1 = e.node1 in value_group
        #    x2 = e.node2 in value_group
        #    if int(x1)+int(x2) == 1:
        #        connects.append(e)
        connects = []
        for e in out_group:
            x1 = e.node1 in value_group
            x2 = e.node2 in value_group
            if int(x1)+int(x2) == 1:
                connects.append(e)
        avg_outgroup = get_avg_edge(connects)
        avg_ingroup = get_avg_edge(in_group)
        #avg_ingroup = last_round
        #avg_outgroup = max_connect
        write_log("\t---add {} {} max_connect_edge={:.4f}/ group_sz = (in: {}/connect: {}/value: {}); score = {:.4f} - {:.4f} = {:.4f}".format(new_node.col_name, old_node.col_name, max_connect, len(in_group), len(connects), len(value_group), avg_ingroup, avg_outgroup, avg_ingroup-avg_outgroup))
        iterstates.append(IterState(avg_ingroup, avg_outgroup, new_node))
        last_round = get_avg_edge(in_group) 
    max_iter = iterstates[0]
    for it in iterstates[:-1]:
        if it.score > max_iter.score:
            max_iter = it

    # early stop criteria: 
    if(max_iter.score <= 0.4): # if previous iteration score is not signification, use the last iteration as the stopping point
        max_iter = iterstates[-1]
    #if max_iter >= TERMINATE_THRESHOLD:
    #    pass
    #elif iterstates[-1].score > INCLUDELAST_THRESHOLD:
    #    max_iter = iterstates[-1].score

    #value_group = set(list(pairs[sorted_y[0][0]]))
    value_group = set([starte.node1, starte.node2])
    for it in iterstates:
        if it.new_node is not None:
            #write_log("\tadd {} / max_out = {:.4f}, min_in = {:.4f}, score = {:.4f}".format(it.new_node.col_name, it.max_connect, it.min_ingroup, it.score))
            value_group.add(it.new_node)
        if it == max_iter:
            break
    _y_predict = [col in value_group for col in table]
    write_log("actual group: {}".format(','.join(filter(lambda x:x is not None, [col.col_name if col.is_value else None for col in table]))))
    write_log("diff: {}".format(','.join(filter(lambda x:x is not None, [col.col_name if col.is_value != (col in value_group) else None for col in table]))))
    write_log("")
    #print("")

    # baseline based on type
    type_count = Counter()
    for col in table:
        type_count.update({str(col.dtype):1})
    copy_cols = [col for col in table]
    random.shuffle(copy_cols)
    values = ('object',1)
    for col in copy_cols:
        if col.is_value:
            values = (col.dtype, 1)
    write_log("baseline group: {}".format(','.join(filter(lambda x:x is not None, [col.col_name if col.dtype == values[0] else None for col in table]))))
    write_log("diff: {}".format(','.join(filter(lambda x:x is not None, [col.col_name if col.is_value != (col.dtype == values[0]) else None for col in table]))))
    flush_log()
    return _y_predict

#y_predict = []
#for table in test:
#    y_predict += get_the_value_group(table, model)    


fps = open('baseline_datafile.txt','w')

baseline_correct_col = []
baseline_recall_col = []
baseline_table = []
model_correct_col = []
model_recall_col = []
model_table = []
fail_reasons = []
k = 0
mask = [False for i in range(len(test))]
start = time.time()
edges = []
recall_avg = []
recall_all = []
prec_avg = []
prec_all = []
for ti,cols in enumerate(test):
    Ncols = len(cols)

    #y_test = y_predict[k:k+Ncols]
    
    write_log("File {} ||| {}".format(cols[0].filename, cols[0].data_index))
    y_test = get_the_value_group(cols, model)
    print("case {}, time = {}".format(ti, time.time()-start))
    if len(y_test) == 0:
        continue
    mask[ti] = True
    model_correct_col.append([cols[i].is_value == y_test[i] for i in range(Ncols)])
    model_recall_col.append([cols[i].is_value == y_test[i] and cols[i].is_value for i in range(Ncols)])
    model_table.append(all([cols[i].is_value == y_test[i] for i in range(Ncols)]))

    count_value = sum([1 if col.is_value else 0 for col in cols])
    count_covered = sum([1 if cols[i].is_value and y_test[i] else 0 for i in range(Ncols)])
    recall_avg.append(float(count_covered) / float(count_value) if count_value > 0 else 1.0)
    recall_all.append((count_covered, count_value))
    prec_avg.append(float(sum([cols[i].is_value==y_test[i] for i in range(Ncols)])) / float(len(y_test))) 
    prec_all.append((float(sum([cols[i].is_value==y_test[i] for i in range(Ncols)])), float(len(y_test)))) 
    for comb in itertools.combinations(range(0,len(cols)), 2):
        truth_ = cols[comb[0]].is_value and cols[comb[1]].is_value
        predict_ = y_test[comb[0]] and y_test[comb[1]]
        edges.append(truth_==predict_)
    
    fps.write("{}||||{}\n".format(cols[0].filename, cols[0].data_index))
    fps.write('predict group = {} \n'.format(','.join(filter(lambda x: x is not None, [str(cols[i].col_name) if y_test[i] else None for i in range(Ncols)]))))
    fps.write('actual group = {}\n'.format(','.join(filter(lambda x: x is not None, [str(cols[i].col_name) if cols[i].is_value else None for i in range(Ncols)]))))
    fps.write('diffs = {} \n\n'.format(','.join(filter(lambda x: x is not None, [str(cols[i].col_name) if cols[i].is_value != y_test[i] else None for i in range(Ncols)]))))

    # baseline based on type
    type_count = Counter()
    for col in cols:
        type_count.update({str(col.dtype):1})
    values = type_count.most_common(1)[0]
    # randomly choose starting point from user
    copy_cols = [col for col in cols]
    random.shuffle(copy_cols)
    for col in copy_cols:
        if col.is_value:
            values = (col.dtype, 1)
    baseline_correct_col.append([cols[i].is_value == (cols[i].dtype == values[0]) for i in range(Ncols)])
    baseline_recall_col.append([cols[i].is_value and cols[i].dtype == values[0] for i in range(Ncols)])
    baseline_table.append(all([cols[i].is_value == (cols[i].dtype == values[0]) for i in range(Ncols)]))
    fail = any([cols[i].is_value != (y_test[i]) for i in range(Ncols)])
    if fail:
        #s = 'predict id = {} / ({})\n'.format(','.join(filter(lambda x: x is not None, [str(cols[i].col_name) if cols[i].dtype == values[0] else None for i in range(Ncols)])), values[0])
        s = 'predict group = {} / ({})\n'.format(','.join(filter(lambda x: x is not None, [str(cols[i].col_name) if y_test[i] else None for i in range(Ncols)])), values[0])
        #s += 'actual id = {}\n'.format(','.join(filter(lambda x: x is not None, [str(cols[i].col_name) if cols[i].is_value else None for i in range(Ncols)])))
        fail_reasons.append((filename_lookup[ti], s))
        write_log(s)
    flush_log()

total_values = float(sum([(sum([c.is_value for c in cols]) if mask[ti] else 0) for ti,cols in enumerate(test)]))
write_log("BASELINE precision = {}".format(float(sum([sum(t) for t in baseline_correct_col])) / float(sum([len(t) for t in baseline_correct_col]))))
write_log("BASELINE recall = {}".format(float(sum([sum(t) for t in baseline_recall_col])) / total_values))
write_log("BASELINE table-wise precision = {}".format(float(sum(baseline_table)) / float(len(test))))
write_log("")
write_log("MODEL precision = {}".format(float(sum([sum(t) for t in model_correct_col])) / float(sum([len(t) for t in model_correct_col]))))
write_log("MODEL recall = {}".format(float(sum([sum(t) for t in model_recall_col])) / total_values))
write_log("MODEL table-wise precision = {}".format(float(sum(model_table)) / float(len(test))))
write_log("avg value col per table = {}".format(float(sum(y_test)) / float(len(test))))

print("edge correct = {} / {} = {}".format(sum(edges), len(edges), float(sum(edges))/float(len(edges))))
print("recall avg = {}".format(sum(recall_avg) / len(recall_avg)))
print("recall all = {}".format(float(sum([rr[0] for rr in recall_all])) / float(sum([rr[1] for rr in recall_all]))))
print("recall avg = {}".format(sum(prec_avg) / len(prec_avg)))
print("recall all = {}".format(float(sum([rr[0] for rr in prec_all])) / float(sum([rr[1] for rr in prec_all]))))
fail_log = open('fail_log.log','w')
for pair in fail_reasons:
    for locate in pair[0]:
        fail_log.write('{}||||{}\n'.format(locate[0], locate[1]))
    fail_log.write('did not predict right\n')
    fail_log.write(pair[1])
    fail_log.write('\n')
