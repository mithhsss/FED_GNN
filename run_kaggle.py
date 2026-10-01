"""
Kaggle Kernel Runner and Monitor.
Pushes FedGATSage kernel to Kaggle, monitors execution, and downloads results.
"""

import os
import sys
import time
import json
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("KaggleRunner")

def monitor_kaggle_kernel(kernel_id: str = 'mythhhhh182678342/fedgatsage-ids-gpu', download_dir: str = 'results_kaggle'):
    api = KaggleApi()
    api.authenticate()
    logger.info("Authenticated with Kaggle API.")
    logger.info(f"Monitoring kernel '{kernel_id}' on Kaggle GPU...")
    
    while True:
        try:
            status_res = api.kernels_status(kernel_id)
            status = str(status_res.status).lower()
            logger.info(f"Current Kernel Status: {status_res.status}")
            
            if 'complete' in status:
                logger.info("Kernel execution completed successfully!")
                break
            elif 'error' in status or 'failed' in status or 'cancel' in status:
                logger.error(f"Kernel execution finished with status: {status_res.status}")
                if hasattr(status_res, 'failure_message') and status_res.failure_message:
                    logger.error(f"Failure message: {status_res.failure_message}")
                break
        except Exception as e:
            logger.warning(f"Status check exception (retrying): {e}")
            
        time.sleep(25)
        
    # Download output artifacts
    os.makedirs(download_dir, exist_ok=True)
    logger.info(f"Downloading kernel output artifacts to {download_dir}...")
    try:
        api.kernels_output(kernel_id, path=download_dir, force=True)
        logger.info(f"Outputs successfully saved to {download_dir}")
        for root, _, files in os.walk(download_dir):
            for f in files:
                logger.info(f"  Artifact: {os.path.join(root, f)}")
    except Exception as e:
        logger.warning(f"Error downloading output: {e}")

if __name__ == '__main__':
    monitor_kaggle_kernel()
