import os
import time
import psutil
import subprocess
import sys

def monitor(pid, limit_gb=100):
    print(f"Monitoring process {pid} with limit {limit_gb}GB")
    process = psutil.Process(pid)
    while True:
        try:
            # Check recursive memory usage of process and children
            mem_info = process.memory_info()
            total_mem = mem_info.rss
            for child in process.children(recursive=True):
                try:
                    total_mem += child.memory_info().rss
                except:
                    pass
            
            total_gb = total_mem / (1024**3)
            if total_gb > limit_gb:
                print(f"\n[MONITOR] Memory usage exceeded {limit_gb}GB ({total_gb:.2f}GB). Killing process...")
                process.kill()
                # Also kill children
                for child in process.children(recursive=True):
                    try:
                        child.kill()
                    except:
                        pass
                sys.exit(1)
            
            # Also check system wide memory to be safe
            sys_mem = psutil.virtual_memory()
            if sys_mem.percent > 90 and (sys_mem.total - sys_mem.available) / (1024**3) > limit_gb:
                print(f"\n[MONITOR] System memory critical ({sys_mem.percent}%). Killing process...")
                process.kill()
                sys.exit(1)
                
            if not process.is_running():
                print("Process finished normally.")
                break
            time.sleep(1)
        except psutil.NoSuchProcess:
            break

if __name__ == "__main__":
    # Command to run passed as arguments
    cmd = sys.argv[1:]
    if not cmd:
        print("Usage: python mem_monitor.py <command>")
        sys.exit(1)
    
    p = subprocess.Popen(cmd)
    monitor(p.pid)
