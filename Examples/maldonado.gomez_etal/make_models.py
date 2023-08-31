import pandas as pd 
import os
import urllib
from pathlib import Path
import gzip
import shutil
from modelseedpy import MSBuilder, MSGenome, RastClient
import cobra as cb
from joblib import Parallel, delayed

def get_genome(taxid,genome_list,refseq_metadata,datapth = '.'):

    """function to download protein fasta file from assembly accession number. Saves to a gzipped faa file in the datapth directory called ``tmp_{Accession ID}.faa.gz``


    :param taxid: NCBI Taxa ID for the taxa we want a model of. Should match the index of the table of genomes we need.
    :type taxid: int

    :param genome_list: Table of information on the genomes we want, indexed by taxa id, including Assembly Accession ID (column name ``Assembly Accession``)
    :type genome_list: pd.DataFrame

    :param refseq_metadata: table of refseq database metadata with FTP path, indexed by assembly accession
    :type refseq_metadata: pd.DataFrame

    :param datapth: desired location to save temp files
    :type datapth: str

    :return: temp file path
    :rtype: str

    """
    assembly_acc = genome_list.loc[taxid,"Assembly Accession"]


    try:
        flurl = refseq_metadata.loc[assembly_acc,"FTP Path"]
    except KeyError:
        flurl = '-'


    #if this is missing we need to peice it together as either
    # ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/XXX/XXX/XXX/WWW_YYYYYYYYY.Z_NNNNNNNNN/
    # or
    # ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/XXX/XXX/XXX/WWW_YYYYYYYYY.Z_NNNNNNNNN/
    # Where the Accession ID is WWW_YYYYYYYYY.Z and XXXXXXXXX == YYYYYYYYY and NNNNNNNN is ????
    # Going to have to parse the webpage to figure it out. While we're at it we can check to make sure a .faa file is there.
    if flurl == '-':
        acc_n_n = assembly_acc.split("_")[-1].split(".")[0]
        attempt1 = "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/{}/{}/{}/".format(acc_n_n[0:3],acc_n_n[3:6],acc_n_n[6:])
        try:
            req = urllib.request.Request(attempt1)
            response = urllib.request.urlopen(req)
            contents = response.read().decode('utf-8')
            flurl =attempt1 + contents.split(" ")[-1].split("\r")[0]
        except:
            attempt2 = "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/{}/{}/{}/".format(acc_n_n[0:3],acc_n_n[3:6],acc_n_n[6:])
            try:
                req = urllib.request.Request(attempt2)
                response = urllib.request.urlopen(req)
                contents = response.read().decode('utf-8')
                flurl =attempt2 + contents.split(" ")[-1].split("\r")[0]
            except Exception as fail:
                print("[get_genome] No FTP URL found")
                return [taxid,None,False,type(fail),"https" + flurl[3:] + "--- FAILED"]

    label = os.path.basename(flurl)
    #get the fasta link - change this to get other formats
    link = os.path.join(flurl,label+'_protein.faa.gz')

    saveto = os.path.join(datapth,"Protein_{}.faa.gz".format(assembly_acc))

    try:
        urllib.request.urlretrieve(link, saveto)
        good = True
        exctp = None
    except Exception as fail:
        print("[get_genome] No protein .faa file at {}".format("https" + flurl[3:]))
        good = False
        exctp = type(fail)

    return [taxid,saveto,good,exctp,"https" + flurl[3:]]

def run_modelseed(TaxID,SeqPath,SeqStatus,SeqDownloadError,FTPLink,savefld = ".",keep_faa = False):

    """Wrapper around modelseedpy to make genome from fasta file and save it

    :param fastafl: path to protein faa file 
    :type fastafl: str

    :param name: desired name for the model
    :type name: str

    :return: COBRA model produced by modelseedpy
    :rtype: cobra model
    """

    if SeqStatus:
        Seq_UZ = SeqPath.replace(".gz","")
        ## Unzip the tmp.fna.gz file
        with gzip.open(SeqPath, 'rb') as f_in:
            with open(Seq_UZ, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(SeqPath)
    else:
        return [TaxID,None,None,SeqStatus,SeqDownloadError,FTPLink]

    genome = MSGenome.from_fasta(Seq_UZ,split = ' ')

    rast = RastClient()
    rast.annotate_genome(genome)

    model = MSBuilder.build_metabolic_model('{}'.format(TaxID), genome)

    save_model = os.path.join(savefld,"{}.xml".format(TaxID))

    cb.io.write_sbml_model(model,save_model)

    if not keep_faa:
        os.remove(Seq_UZ)

    return [TaxID,save_model,None,SeqStatus,SeqDownloadError,FTPLink]


def make_all_models(genome_list,savefld = ".",nj_dl = 1,nj_mod = 1,keep_faa = False):
    """
    For each taxa in the the list of genomes (which must include column for Assembly Accession), download the FASTA file from RefSeq, unzip, build cobra model, and save.

    :param taxid: NCBI Taxa ID for the taxa we want a model of. Should match the index of the table of genomes we need.
    :type taxid: int

    :param genome_list: Table of information on the genomes we want, indexed by taxa id, including Assembly Accession ID (column name ``Assembly Accession``)
    :type genome_list: pd.DataFrame

    :param savefld: directory to save the model in, as well as where to store temporary faa files.
    :type savefld: str

    :return: Table of model info and locations
    :rtype: pd.DataFrame

    """
    save_model = os.path.join(savefld,"xml_models")
    Path(save_model).mkdir(parents=True, exist_ok=True)
    save_faa = os.path.join(savefld,"protein_seqs")
    Path(save_faa).mkdir(parents=True,exist_ok=True)

    refseq_metadata = pd.read_csv("https://ftp.ncbi.nlm.nih.gov/genomes/GENOME_REPORTS/prokaryotes.txt",sep = '\t',low_memory=False)
    refseq_metadata.index = refseq_metadata['Assembly Accession'].values

    faa_file_info = Parallel(n_jobs = nj_dl)(delayed(get_genome)(tid,genome_list,refseq_metadata,datapth=save_faa) for tid in genome_list.index)

    faa_info = pd.DataFrame(faa_file_info,columns = ["TaxID","SeqPath","SeqStatus","SeqDownloadError","FTPLink"])
    faa_info.index = faa_info["TaxID"]

    make_models = Parallel(n_jobs=nj_mod)(delayed(run_modelseed)(*faa_info.loc[tid].values,savefld=save_model,keep_faa = keep_faa) for tid in genome_list.index)

    if not keep_faa:
        os.removedirs(save_faa)

    model_info = pd.DataFrame(make_models,columns = ["TaxID","ModelPath","SeqPath","SeqDownloadStatus","SeqDownloadError","FTPLink"])
    model_info.index = model_info["TaxID"]


    return model_info

if __name__ == "__main__":

    datapth = "b_longum_data"

    models_needed = pd.read_csv(os.path.join(datapth,"refseq_genomes.csv"),index_col = 0)

    model_info = make_all_models(models_needed,savefld=os.path.join(datapth,"GSMs"),nj_dl=-1,nj_mod = 3)

    model_info.to_csv(os.path.join(datapth,"model_list.csv"))