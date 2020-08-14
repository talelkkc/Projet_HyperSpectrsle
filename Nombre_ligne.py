#fic=open('spec125.txt', 'r')
def _Nmb_ligne(file):
    fic=open(file, 'r')
    lecture = fic.readlines()
    N_lignes = len(lecture)
    fic.close()
    return  N_lignes