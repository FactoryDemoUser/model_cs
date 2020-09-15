import numpy as np
import pandas as pd
from os import listdir
from scipy.stats import f as f_distrib
from numpy import linalg as la
import statistics
import itertools
import copy
from sklearn.preprocessing import MinMaxScaler
import gc





class Model():
    

    def __init__(self):
        self.name = 'climate_system_model_ivolga'
        self.sink_columns = ['dt', 'upload_id', 'stock_num', 'wagon_id']
        self.sensors = ['HI']
        self.common_params = []
        
        
        

    def get_query(self):
        query = """
            SELECT DISTINCT
                {dt} as dt,
                {upload_id} as upload_id,
                {stock_num} as stock_num,
                {value_float} as value_float,
                {value_str} as value_str,
                {sensor_id} as sensor_id
            FROM {db_name}.{table_name}   
            WHERE {upload_id} = '{upload_id_val}' AND {stock_num} = '{stock_num_val}' AND 
            (match(sensor_id, 'SOC_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]_CLIMAT_IN_TEMP') > 0 OR
            match(sensor_id, 'SOC_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]_CLIMAT_MODE') > 0 OR
            match(sensor_id, 'SOC_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]_CLIMAT_SUBMODE[1, 2]') > 0 OR
            match(sensor_id, 'car_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]_BCU_out_Reference_Speed') > 0)
            ORDER BY {dt}
        """
        return query
  

        
        
    def check_sensors(self, df, wagon_id):
        cols = set(df.columns)
        cols_w = set([
            f'SOC_{wagon_id}_CLIMAT_MODE',
            f'SOC_{wagon_id}_CLIMAT_SUBMODE1',
            f'SOC_{wagon_id}_CLIMAT_SUBMODE2',
            f'car_{wagon_id}_BCU_out_Reference_Speed',
            f'car_{wagon_id}_BCU_out_Reference_Speed',
            f'SOC_{wagon_id}_CLIMAT_IN_TEMP'])

        return len(cols_w) != len(cols.intersection(cols_w))
    
    
    
    
    def predict(self, df):
        if len(df) == 0:
            return pd.DataFrame(columns=self.sink_columns + self.sensors)
        
        stock_num = df['stock_num'].values[0]
        upload_id = df['upload_id'].values[0]
        dt = df['dt'].values

        sensors = df.sensor_id.unique()
        matchers = ['MODE','SUBMODE1', 'SUBMODE2']
        matching = [s for s in sensors if any(xs in s for xs in matchers)]

        df_float = df[~df['sensor_id'].isin(matching)]
        df_str = df[df['sensor_id'].isin(matching)]
        df_float['value_float'] = df_float['value_float'].astype(np.float32)#.replace(np.nan, 999999)???
        #df_str['value_str'] = df_str['value_str'].replace(np.nan, 999999)???
        cols_str = ['NA', 'PRE_HEATING', 'PRE_COOLING', 
                    'COOLING', 'VENTING', 'HEATING', 
                    'OFF', 'PRE', 'SAVE', 'AUTO']
        df_str['value'] = df_str['value_str'].replace(cols_str, range(1,11)).astype(np.uint8)

        df_f = df_float.pivot_table(index = df_float.index, columns='sensor_id', values='value_float')
        df_s = df_str.pivot_table(index = df_str.index, columns='sensor_id', values='value') 

        df = pd.concat([df_f, df_s], axis=1)
        #df = df.replace(999999, np.nan)???

        df['dt'] = dt
        df = df.fillna(method = 'ffill')
        df = df.dropna()
        df.columns.name = ''

        del df_f, df_s
        gc.collect()
        gc.collect()
        
        result = []
        for i in range(1,12):
            if self.check_sensors(df, i):
                continue

            cond_wag = ((df[f'SOC_{i}_CLIMAT_MODE'] >= 10) &
                        (df[f'SOC_{i}_CLIMAT_SUBMODE1'] >= 4) &
                        (df[f'SOC_{i}_CLIMAT_SUBMODE2'] >= 4) &
                        (df[f'car_{i}_BCU_out_Reference_Speed'] >= 140) & 
                        (df[f'SOC_{i}_CLIMAT_IN_TEMP'] <= 40))

            col_t = f'SOC_{i}_CLIMAT_IN_TEMP'
            result_w = df.loc[cond_wag, ['dt', col_t]]
            result_w['wagon_id'] = i
            result_w['HI'] = 1

            result_w.loc[(result_w[col_t]<15) | (result_w[col_t]>25), 'HI'] = 0.5
            result_w.drop(labels=col_t, axis =1, inplace=True)
            result.append(result_w)

        if len(result)==0:
            return pd.DataFrame(columns=self.sink_columns + self.sensors)
            
        gc.collect()
        result = pd.concat(result)
        result['stock_num'] = stock_num
        result['upload_id'] = upload_id
        
        return result