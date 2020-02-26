import pandas as pd
import subprocess
import os


def load_TCGA_manifest():
    manifest_path = os.environ['MANIFEST_PATH']
    df = pd.read_csv(manifest_path, sep='\t')
    return df


def create_manifest_for_one_patient(dataframe, case_id):
    manifest_path = os.environ['MANIFEST_PATH']
    single_manifest_path = os.path.join(manifest_path, 'manifest_{}.txt'.format(case_id))
    dataframe[dataframe.id == case_id].to_csv(single_manifest_path, header=True, index=None, sep='\t')
    return None


def download_from_manifest(single_manifest_path):
    subprocess.run([os.environ['GDC_PATH'], 'download', '-m', single_manifest_path])
    return None


def download_from_case_id(case_id):
    subprocess.run([os.environ['GDC_PATH'], 'download', case_id])
    return None


def sync_with_s3(case_id, s3_TCGA_bucket="s3://evotec/pathomix/data/TCGA", cohort_name="COAD"):
    source_path = os.path.join(os.environ['PATHOMIX_DATA'], "TCGA", case_id)
    target_path = os.path.join(s3_TCGA_bucket, cohort_name, case_id)
    subprocess.run(["aws", "s3", "sync", source_path, target_path])
    return None


def delete_TCGA_folder(case_id):
    source_path = os.path.join(os.environ['PATHOMIX_DATA'], "TCGA", case_id)
    subprocess.run("rm", "-dr", source_path)
    return None


if __name__=="__main__":
    total_manifest_path = ""
    df = load_TCGA_manifest()
    for case_id in df.id:
        download_from_case_id(case_id)
        sync_with_s3(case_id)
        delete_TCGA_folder(case_id)
