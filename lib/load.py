import yaml
import itertools
import numpy as np
import pandas as pd
import json

def load_file(file: str)->dict:
    file = open(file)
    d_f = yaml.safe_load(file)
    for k in d_f:
        d_f[k]['c_u'].append([0,0])
        if 'postpone' not in d_f[k]:
            d_f[k]['postpone'] = 0
        if 'mandatory' not in d_f[k]:
            d_f[k]['mandatory'] = False
    return d_f

class LoadData:
    @staticmethod
    def get_cost_value_form(data: list)->dict:
        def sort_by_cost(arr):
            return arr[0]
        for item in data:
            item['c_u'] = [sorted(it,key=sort_by_cost) for it in item['c_u']]
            for index in range(len(item['c_u'])):
                item['c_u'][index] = [[i]+it for i, it in enumerate(item['c_u'][index])]
        y_con_goals = {}
        y_strategy = {}
        for item in data:
            y_con_goals[item['t']] = []
            y_strategy[item['t']] = []
            for it in itertools.product(*item['c_u']):
                np_arr = np.array(it)
                stra, c_g = np_arr[:,:1],np_arr[:,1:] 
                c_g = np.sum(c_g, axis=0)
                y_strategy[item['t']].append(stra.flatten().tolist())
                y_con_goals[item['t']].append(c_g.tolist())
        return data, y_strategy, y_con_goals

    @staticmethod        
    def get_k_form(strategy: dict, c_u: dict)->dict:

        def sort_by_cost(a):
            return a['c_u'][0]

        def get_k_items(strategy, c_u):
            s_c_u = {}
            for y in strategy:
                s_c_u[y] = [{'strategy': strategy[y][index], 'c_u': c_u[y][index]} for index in range(len(strategy[y]))]
                s_c_u[y].sort(key=sort_by_cost)
                tmp = []
                for item in s_c_u[y]:
                    if len(tmp) == 0:
                        tmp.append(item)
                        continue
                    if tmp[-1]['c_u'][1] >= item['c_u'][1]:
                        continue
                    if tmp[-1]['c_u'][0] == item['c_u'][0]:
                        tmp.pop()
                    tmp.append(item)
                s_c_u[y] = tmp
            return s_c_u

        def get_kstrategy_cost_utility(k_items):
            ks = {}
            cost_v= {}
            util_v = {}
            for y in k_items:
                ks[y] = [ item['strategy'] for item in k_items[y] ]
                cost_v[y] = [item['c_u'][0] for item in k_items[y]]
                util_v[y] = [item['c_u'][1] for item in k_items[y]]

            return ks, cost_v, util_v

        k_items = get_k_items(strategy=strategy, c_u=c_u)

        return get_kstrategy_cost_utility(k_items)
        
    @staticmethod            
    def load(file: str)->dict:
        def get_y_goals(data: dict)->dict:
            df = pd.DataFrame(data)
            df = df.groupby(['t'])['c_u'].apply(lambda x: x.tolist()).reset_index()
            return json.loads(df.to_json(orient = "records"))  

        raw = list(load_file(file).values())
        y_goals = get_y_goals(raw)
        data, y_strategy, y_c_u = LoadData.get_cost_value_form(y_goals)
        # print(LoadData.get_k_form(y_strategy, y_c_u))
        return data, *LoadData.get_k_form(y_strategy, y_c_u)

class LoadData2:
    @staticmethod
    def load(file: str) -> dict:
        raw = list(load_file(file).values())
        res_o = {}
        res = []
        max_t = -1
        for item in raw:
            if item['t'] not in res_o:
                res_o[item['t']] = []
            res_o[item['t']].append(item)
            if max_t < item['t']:
                max_t = item['t']
        for i in range(max_t+1):
            if i in res_o:
                res.append(res_o[i])
            else:
                res.append([])
        return res

LoadData.load("asset/new_user_goal.yml")