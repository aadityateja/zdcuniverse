import pandas as pd
import numpy as np

print("Loading files...")

parent = pd.read_parquet("../data/parent.parquet")
zdc = pd.read_parquet("../data/zdc.parquet")

print("Parent events:", len(parent))
print("ZDC events:", len(zdc))

features = []

for i in range(len(parent)):

    truth_energy = parent.iloc[i]["MCParticles.momentum"][0]

    hcal_edep = zdc.iloc[i]["HcalFarForwardZDCHits.eDep"]
    wsi_edep = zdc.iloc[i]["ZDC_WSi_Hits.eDep"]

    hcal_energy = np.sum(hcal_edep)
    wsi_energy = np.sum(wsi_edep)

    total_energy = hcal_energy + wsi_energy

    num_hcal_hits = len(hcal_edep)
    num_wsi_hits = len(wsi_edep)

    features.append({
        "event_id": i,
        "truth_energy": truth_energy,
        "hcal_energy": hcal_energy,
        "wsi_energy": wsi_energy,
        "total_energy": total_energy,
        "num_hcal_hits": num_hcal_hits,
        "num_wsi_hits": num_wsi_hits
    })

df = pd.DataFrame(features)

print("\nFirst 5 events:")
print(df.head())

print("\nAverage deposited energy:")
print(df["total_energy"].mean())

print("\nAverage missing energy:")
print(
    df["truth_energy"].mean()
    -
    df["total_energy"].mean()
)

df.to_csv("event_features.csv", index=False)

print("\nSaved event_features.csv")
