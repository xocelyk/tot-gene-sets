import pandas as pd
import gseapy as gp

dbs = gp.get_library_name()

def get_GO_Enrich(systemGenes):
    systemGenes = systemGenes.split(', ')
    enr = gp.enrichr(gene_list=systemGenes, # or "./tests/data/gene_list.txt",
                 gene_sets=['GO_Biological_Process_2023'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
    top_terms = []
    for i in range(3):
        top_terms.append(enr.results.iloc[i]['Term'])
    return top_terms