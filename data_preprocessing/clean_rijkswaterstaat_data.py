#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script modifies the tidal gauge data csv files obtained from Rijkswaterstaat to a simpler csv file format. It mainly removes unecessary information.


Created on Mon Apr 11 15:01:26 2022

@author: Matias Duran-Matute (m.duran.matute@tue.nl)
"""

#%%
import pandas as pd
from datetime import datetime

#%%

#custom_date_parser = lambda x: datetime.strptime(x, "%d-%m-%Y %H:%M")

D = pd.read_csv('../data_raw/Harling_raw.csv', delimiter=";", parse_dates=[["datum", 'tijd']], dayfirst=True)#, date_parser=custom_date_parser)

#%%
d = {'time': D.datum_tijd, 'level': D.waarde}
df = pd.DataFrame(data=d)

df.to_csv('../data/Harling.csv')
