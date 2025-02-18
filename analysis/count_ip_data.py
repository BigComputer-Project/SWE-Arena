import os
import logging
from datetime import datetime, timedelta
from urllib.parse import unquote
import json
from collections import defaultdict
import smbclient
import shutil
import re

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.disabled = True

def get_ip_from_jsonl(file_path):
    """Extract IP from the first line of a JSONL file"""
    try:
        with smbclient.open_file(file_path, mode='r') as f:
            first_line = f.readline()
            data = json.loads(first_line)
            return data.get('ip')
    except Exception as e:
        log.error(f"Error reading file {file_path}: {e}")
        return None

def get_chat_session_id(file_path):
    """Extract chat_session_id based on the file location:
    - For files under conv_logs: extract from filename
    - For files under sandbox_logs: read from file content
    """
    try:
        if 'conv_logs' in file_path:
            # Extract from filename for conv_logs
            # Handle Windows UNC path format
            filename = file_path.split('\\')[-1]  # Get the last component of the path
            match = re.match(r'conv-log-([a-f0-9]+)\.json', filename)
            if match:
                return match.group(1)
        elif 'sandbox_logs' in file_path:
            # Read from file content for sandbox_logs
            with smbclient.open_file(file_path, mode='r') as f:
                data = json.loads(f.read())
                return data['sandbox_state'].get('chat_session_id')
        return None
    except Exception as e:
        log.error(f"Error getting chat_session_id from {file_path}: {e}")
        return None

def get_sandbox_session_ids(server, share, date_str):
    """Get all chat_session_ids from sandbox logs for a given date"""
    sandbox_folder = f"\\\\{server}\\{share}\\{date_str}\\sandbox_logs"
    session_ids = set()
    
    if not smbclient.path.exists(sandbox_folder):
        return session_ids
    
    try:
        for file_info in smbclient.scandir(sandbox_folder):
            if file_info.name.endswith('.json'):
                file_path = f"{sandbox_folder}\\{file_info.name}"
                session_id = get_chat_session_id(file_path)
                if session_id:
                    session_ids.add(session_id)
    except Exception as e:
        log.error(f"Error scanning sandbox folder {sandbox_folder}: {e}")
    
    return session_ids

def count_files_per_ip(smb_url, start_date_str="2025_02_18"):
    """Count files per IP address from the given start date"""
    # Remove 'smb://' prefix and parse URL components
    url = smb_url[6:]
    creds_server, share = url.split('/', 1)
    creds, server = creds_server.rsplit('@', 1)
    username, password = creds.split(':', 1)
    password = unquote(password)
    
    # Register the SMB session
    smbclient.register_session(server, username=username, password=password)
    
    # Convert start date string to datetime
    start_date = datetime.strptime(start_date_str, "%Y_%m_%d")
    ip_counts = defaultdict(int)
    
    try:
        # Get current date for iteration
        current_date = start_date
        today = datetime.now()
        
        while current_date <= today:
            date_str = current_date.strftime("%Y_%m_%d")
            folder_path = f"\\\\{server}\\{share}\\{date_str}\\conv_logs\\battle_anony"
            
            try:
                # List all JSON files in the battle_anony folder
                if smbclient.path.exists(folder_path):
                    for file_info in smbclient.scandir(folder_path):
                        if file_info.name.endswith('.json'):
                            file_path = f"{folder_path}\\{file_info.name}"
                            ip = get_ip_from_jsonl(file_path)
                            if ip:
                                ip_counts[ip] += 1
            except Exception as e:
                log.error(f"Error processing folder {date_str}: {e}")
            
            # Move to next day
            current_date += timedelta(days=1)
                
    except Exception as e:
        log.error(f"Error accessing SMB share: {e}")
    
    return dict(ip_counts)

def download_files_by_ip(smb_url, start_date_str="2025_02_18"):
    """Download files and organize them by IP address"""
    # Remove 'smb://' prefix and parse URL components
    url = smb_url[6:]
    creds_server, share = url.split('/', 1)
    creds, server = creds_server.rsplit('@', 1)
    username, password = creds.split(':', 1)
    password = unquote(password)
    
    # Register the SMB session
    smbclient.register_session(server, username=username, password=password)
    
    # Create base data directory
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Convert start date string to datetime
    start_date = datetime.strptime(start_date_str, "%Y_%m_%d")
    
    try:
        # Get current date for iteration
        current_date = start_date
        today = datetime.now()
        
        while current_date <= today:
            date_str = current_date.strftime("%Y_%m_%d")
            folder_path = f"\\\\{server}\\{share}\\{date_str}\\conv_logs\\battle_anony"
            
            # Get all sandbox session IDs for this date
            sandbox_session_ids = get_sandbox_session_ids(server, share, date_str)
            try:
                # List all JSON files in the battle_anony folder
                if smbclient.path.exists(folder_path):
                    for file_info in smbclient.scandir(folder_path):
                        if file_info.name.endswith('.json'):
                            file_path = f"{folder_path}\\{file_info.name}"
                            ip = get_ip_from_jsonl(file_path)
                            if ip:
                                # Create directory structure for this IP
                                ip_dir = os.path.join(data_dir, ip)
                                valid_dir = os.path.join(ip_dir, "valid")
                                invalid_dir = os.path.join(ip_dir, "invalid")
                                os.makedirs(valid_dir, exist_ok=True)
                                os.makedirs(invalid_dir, exist_ok=True)
                                
                                # Check if chat_session_id exists in sandbox logs
                                chat_session_id = get_chat_session_id(file_path)
                                has_sandbox = chat_session_id in sandbox_session_ids if chat_session_id else False
                                print(chat_session_id, has_sandbox)
                                # Determine target directory based on sandbox log existence
                                target_dir = valid_dir if has_sandbox else invalid_dir
                                
                                # Download the file
                                local_file_path = os.path.join(target_dir, file_info.name)
                                try:
                                    with smbclient.open_file(file_path, mode='rb') as remote_file:
                                        with open(local_file_path, 'wb') as local_file:
                                            shutil.copyfileobj(remote_file, local_file)
                                    log.info(f"Downloaded {file_info.name} to {target_dir}")
                                except Exception as e:
                                    log.error(f"Error downloading file {file_info.name}: {e}")
            
            except Exception as e:
                log.error(f"Error processing folder {date_str}: {e}")
            
            # Move to next day
            current_date += timedelta(days=1)
                
    except Exception as e:
        log.error(f"Error accessing SMB share: {e}")

def main():
    smb_url = os.getenv("SMB_URL")
    
    # Download files organized by IP
    print("\nDownloading files and organizing by IP address...")
    download_files_by_ip(smb_url)
    
    # Count and display statistics
    ip_counts = count_files_per_ip(smb_url)
    print("\nFile counts per IP address:")
    for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"IP: {ip:<15} Count: {count}")

if __name__ == "__main__":
    main()