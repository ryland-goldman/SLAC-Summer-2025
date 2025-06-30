download_files = [
    {"name":"downloadedscript.py","url":"https://raw.githubusercontent.com/ryland-goldman/SLAC-Summer-2025/refs/heads/main/Tungsten-Target-EnergyThicknessMeasurement/batch_cloud.py"},
    {"name":"TungstenTarget.g4bl","url":"https://raw.githubusercontent.com/ryland-goldman/SLAC-Summer-2025/refs/heads/main/Tungsten-Target-EnergyThicknessMeasurement/TungstenTarget.g4bl"}
]

import requests
for file in download_files:
    response = requests.get(file["url"])
    with open(file["name"],"w") as f: f.write(response.text)

import downloadedscript
