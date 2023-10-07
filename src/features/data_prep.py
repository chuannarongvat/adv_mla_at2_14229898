import pandas as pd
import numpy as np

class MeltDataFrame:
    def __init__(self, df, id_vars, value_vars_start, value_vars_end, var_name, value_name):
        self.df = df
        self.id_vars = id_vars
        self.value_vars_start = value_vars_start
        self.value_vars_end = value_vars_end
        self.var_name = var_name
        self.value_name = value_name
    
    def melt(self):
        melted_df = pd.melt(
            self.df,
            id_vars=self.id_vars,
            value_vars=[f'd_{i}' for i in range(self.value_vars_start, self.value_vars_end)],
            var_name=self.var_name,
            value_name=self.value_name
        )
        return melted_df