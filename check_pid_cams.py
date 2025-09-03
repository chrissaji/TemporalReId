import os
import re
from collections import defaultdict

# Paths to query and gallery image folders
query_dir = os.path.join('data', 'VeRi', 'image_query')
gallery_dir = os.path.join('data', 'VeRi', 'image_test')

def parse_pid_cam(filename):
    # Example: 0002_c002_00030600_0.jpg
    m = re.match(r'(\d+)_c(\d+)_', filename)
    if m:
        pid = m.group(1)
        camid = m.group(2)
        return pid, camid
    return None, None

def collect_pid_cam(folder):
    pid2cams = defaultdict(set)
    for fname in os.listdir(folder):
        if fname.endswith('.jpg'):
            pid, camid = parse_pid_cam(fname)
            if pid and camid:
                pid2cams[pid].add(camid)
    return pid2cams

if __name__ == '__main__':
    print('Scanning query set...')
    query_pid2cams = collect_pid_cam(query_dir)
    print('Scanning gallery set...')
    gallery_pid2cams = collect_pid_cam(gallery_dir)

    # Example: print for PID 0002
    pid = '0002'
    print(f'Query cameras for PID {pid}:', sorted(query_pid2cams.get(pid, [])))
    print(f'Gallery cameras for PID {pid}:', sorted(gallery_pid2cams.get(pid, [])))

    # Optionally, print all PIDs with their camera sets
    # for pid in sorted(query_pid2cams):
    #     print(f'PID {pid} - Query cams: {sorted(query_pid2cams[pid])}, Gallery cams: {sorted(gallery_pid2cams.get(pid, []))}')
