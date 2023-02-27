# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 20:57:24 2022

@author: zhang
"""

#directly download the zip from the internet 
#unzip the data into local folder
import requests
import zipfile
import os
print(os.path.abspath(os.curdir))
os.chdir("../..")
os.chdir(r'data/raw')
print(os.path.abspath(os.curdir))

def download(url,name):

    r=requests.get(url)
    with open(name,'wb') as f:
        f.write(r.content)

    with zipfile.ZipFile(name,'r') as myfile:
        myfile.extractall()
        
#url='https://github.com/CapitalOneRecruiting/DS/archive/refs/heads/master.zip'
#name='transactions.zip'

download(url,name)