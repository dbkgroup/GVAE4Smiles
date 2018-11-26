# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:05:15 2018

@author: Steve O'Hagan
"""
import getData as GD

fn = 'data/250kChEMBL23'

smi = GD.getSmi(fn)

good,bad = GD.filterOK(smi)

GD.saveSmi(good,'data/cleanChEMBL23')

GD.saveSmi(bad,'data/badChEMBL23')

