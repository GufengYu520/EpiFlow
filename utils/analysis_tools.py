from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np



def calculate_instability(peptides):
    instability_scores = []
    for peptide in peptides:
        try:
            analysis = ProteinAnalysis(peptide)
            instability = analysis.instability_index()
            instability_scores.append(instability)
        except KeyError:
            instability_scores.append(np.nan)
    return instability_scores
