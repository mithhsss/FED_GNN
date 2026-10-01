import os
import sys
import re
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def fetch_kernel_results(token, kernel_full, target_dir, skip_checkpoints=True):
    os.makedirs(target_dir, exist_ok=True)
    os.environ['KAGGLE_API_TOKEN'] = token
    api = KaggleApi()
    api.authenticate()
    
    owner_slug, kernel_slug, _ = api.parse_kernel_string(kernel_full)
    print(f"\nFetching outputs for {kernel_full} into {target_dir}...")
    
    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner_slug
        req.kernel_slug = kernel_slug
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
    
    # Save log with utf-8
    if resp.log:
        log_path = os.path.join(target_dir, f"{kernel_slug}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(resp.log)
        print(f"  [+] Log saved: {log_path} ({len(resp.log):,} chars)")
    
    # Download files
    for item in resp.files or []:
        fname = item.file_name
        if skip_checkpoints and fname.endswith(".pt"):
            print(f"  [-] Skipping large checkpoint: {fname}")
            continue
        
        out_path = os.path.join(target_dir, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"  [+] Downloading {fname}...")
        r = requests.get(item.url, stream=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        print(f"      Saved: {out_path} ({os.path.getsize(out_path):,} bytes)")

if __name__ == '__main__':
    # Kovuri V1
    fetch_kernel_results("KGAT_1ada211c784deb430fefff5e6611347e",
                         "kovurimithul/fedgatsage-ids-nf-v1",
                         "kaggle_results_v1_kovuri",
                         skip_checkpoints=True)

    # Myth Main V1
    fetch_kernel_results("KGAT_3dd371dbc120251eecce73f10a09f345",
                         "mythhhhh182678342/fedgatsage-ids-gpu",
                         "kaggle_results_v1_main",
                         skip_checkpoints=True)
