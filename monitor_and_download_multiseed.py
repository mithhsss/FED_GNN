import os
import sys
import time
import json
import glob
import numpy as np
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

TOKEN = "KGAT_3dd371dbc120251eecce73f10a09f345"
KERNEL_SLUG = "mythhhhh182678342/fedgatsage-ids-gpu"
TARGET_DIR = "kaggle_results_multiseed"

def monitor_kernel():
    os.environ['KAGGLE_API_TOKEN'] = TOKEN
    api = KaggleApi()
    api.authenticate()
    
    print("=" * 90)
    print(f"Monitoring Live Kaggle Kernel: {KERNEL_SLUG}")
    print("=" * 90)
    
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            status_info = api.kernels_status(KERNEL_SLUG)
            status = getattr(status_info, 'status', str(status_info))
            msg = getattr(status_info, 'failure_message', None)
            
            elapsed = (time.time() - start_time) / 60.0
            print(f"[{elapsed:5.1f}m elapsed] Status: {status}" + (f" | Msg: {msg}" if msg else ""), flush=True)
            
            status_str = str(status).lower()
            if 'complete' in status_str:
                print("\n[+] Kernel execution completed successfully!", flush=True)
                break
            elif 'error' in status_str or 'cancel' in status_str:
                print(f"\n[-] Kernel ended with status: {status}", flush=True)
                break
        except Exception as e:
            print(f"Polling warning: {e}", flush=True)
            
        time.sleep(30)

def download_outputs():
    os.environ['KAGGLE_API_TOKEN'] = TOKEN
    os.makedirs(TARGET_DIR, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    
    owner_slug, kernel_name, _ = api.parse_kernel_string(KERNEL_SLUG)
    print(f"\nDownloading outputs for {KERNEL_SLUG} into {TARGET_DIR}...")
    
    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner_slug
        req.kernel_slug = kernel_name
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        
    if resp.log:
        log_path = os.path.join(TARGET_DIR, f"{kernel_name}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(resp.log)
        print(f"  [+] Log saved: {log_path} ({len(resp.log):,} chars)")
        
    for item in resp.files or []:
        fname = item.file_name
        if fname.endswith(".pt"):
            continue
        out_path = os.path.join(TARGET_DIR, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"  [+] Downloading {fname}...")
        r = requests.get(item.url, stream=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        print(f"      Saved: {out_path} ({os.path.getsize(out_path):,} bytes)")

def print_final_report():
    print("\n" + "=" * 100)
    print("AUDITING & AGGREGATING INDIVIDUAL RUN JSON FILES")
    print("=" * 100)
    
    json_files = sorted(glob.glob(os.path.join(TARGET_DIR, "run_*_seed*.json")))
    if not json_files:
        # Check subdirectories
        json_files = sorted(glob.glob(os.path.join(TARGET_DIR, "**", "run_*_seed*.json"), recursive=True))
        
    print(f"Total Individual Run Files Found: {len(json_files)}")
    for jf in json_files:
        print(f"  [FILE] {os.path.basename(jf)}")
        
    records = []
    for jf in json_files:
        with open(jf, 'r') as f:
            records.append(json.load(f))
            
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['config'], r['split'])].append(r)
        
    PAPER_8 = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']
    
    print("\n" + "=" * 140)
    print(f"{'Config':<28} {'Split':<18} {'Balanced Acc (%)':<18} {'Macro F1 (%)':<18} {'Backdoor Rec (%)':<18} {'Scanning Rec (%)':<18} {'Password Rec (%)':<18}")
    print("=" * 140)
    
    for (cfg, split), recs in grouped.items():
        b_accs = [r['balanced_accuracy'] * 100 for r in recs]
        mf1s = [r['macro_f1'] * 100 for r in recs]
        bd_recs = [r['per_class']['Backdoor']['recall'] * 100 for r in recs]
        sc_recs = [r['per_class']['Scanning']['recall'] * 100 for r in recs]
        pw_recs = [r['per_class']['Password']['recall'] * 100 for r in recs]
        
        b_acc_str = f"{np.mean(b_accs):.2f} ± {np.std(b_accs):.2f}"
        mf1_str = f"{np.mean(mf1s):.2f} ± {np.std(mf1s):.2f}"
        bd_str = f"{np.mean(bd_recs):.2f} ± {np.std(bd_recs):.2f}"
        sc_str = f"{np.mean(sc_recs):.2f} ± {np.std(sc_recs):.2f}"
        pw_str = f"{np.mean(pw_recs):.2f} ± {np.std(pw_recs):.2f}"
        
        print(f"{cfg:<28} {split:<18} {b_acc_str:<18} {mf1_str:<18} {bd_str:<18} {sc_str:<18} {pw_str:<18}")
    print("=" * 140)

if __name__ == '__main__':
    monitor_kernel()
    download_outputs()
    print_final_report()
