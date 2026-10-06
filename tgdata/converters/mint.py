"""The 84 MiNT ERC20 token transaction networks (Shamsi, Ngo et al., 2025), from the Zenodo
release of their edge lists and Edge Growth/Shrink labels.

Each network is a discrete-time graph of weekly windows of Ethereum transfers that advance one
day at a time: snapshot t holds every transfer of days t-1 .. t+5 after the first day, so each
transfer recurs in up to 7 consecutive snapshots. The release is stored as is: one row per
transfer and snapshot, all multi-edges, weights as token amounts, and the graph label of each
snapshot (`y["edge_gs"]`). The 64 `train` and 20 `test` networks are the paper's.

python -m tgdata.converters.mint --edgelists MiNT_edgelists.zip --labels MiNT_labels_Edge_GS.zip \
    --out DIR [--token 0x0 ...] [--push]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

from .. import hub
from .. import io as tgio
from ..schema import Split, Target, TemporalGraph, validate

ZENODO = "https://zenodo.org/records/15364297"
CODE = "github.com/benjaminnNgo/ScalingTGNs"
DAY = 86400
WINDOW = 7
COLUMNS = ["source", "destination", "weight", "date", "snapshot"]
CITATION = """
@inproceedings{shamsi2025mint,
  title={{MiNT}: Multi-Network Transfer Benchmark for Temporal Graph Learning},
  author={Shamsi, Kiarash and Ngo, Tran Gia Bao and Shirzadkhani, Razieh and Huang, Shenyang and
          Poursafaei, Farimah and Azad, Poupak and Rabbany, Reihaneh and Coskunuzer, Baris and
          Rabusseau, Guillaume and Akcora, Cuneyt Gurcan},
  booktitle={Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
  year={2025}
}
"""

TRAIN = (  # dataset_package_64.txt; the first k are its scaling packages (2, 4, ... 32)
    "IOTX", "NOIA", "QSP", "RGT", "CMT", "Yf-DAI", "Mog", "FEG", "PUSH", "HOP", "RSR", "ORN",
    "SLP", "SUPER", "ALBT", "sILV2", "DODO", "BOB", "GHST", "YFII", "aDAI", "DRGN", "CELR",
    "POOH", "TVK", "PICKLE", "LINA", "cDAI", "ANT", "TNT", "WOOL", "BITCOIN", "LQTY", "AUDIO",
    "RLB", "SWAP", "OHM", "RARI", "REP", "LADYS", "BTRFLY", "bendWETH", "TURBO", "0x0", "PRE",
    "AIOZ", "crvUSD", "CRU", "MIM", "SPONGE", "aUSDC", "KP3R", "STARL", "ShibDoge", "LUSD",
    "PSYOP", "steCRV", "MAHA", "POLS", "RFD", "INU", "DPI", "ARC", "INJ",
)
TEST = (  # dataset_package_test.txt
    "MIR", "DOGE2.0", "MUTE", "EVERMOON", "DERC", "ADX", "HOICHI", "SDEX", "BAG", "XCN",
    "ETH2x-FLI", "stkAAVE", "GLM", "QOM", "WOJAK", "DINO", "Metis", "REPv2", "TRAC", "BEPRO",
)
ADDRESS = {
    "IOTX": "0x6fb3e0a217407efff7ca062d46c26e5d60a14d69",
    "NOIA": "0xa8c8cfb141a3bb59fea1e2ea6b79b5ecbcd7b6ca",
    "QSP": "0x99ea4db9ee77acd40b119bd1dc4e33e1c070b80d",
    "RGT": "0xd291e7a03283640fdc51b121ac401383a46cc623",
    "CMT": "0xf85feea2fdd81d51177f6b8f35f0e6734ce45f5f",
    "Yf-DAI": "0xf4cd3d3fda8d7fd6c5a500203e38640a70bf9577",
    "Mog": "0xaaee1a9723aadb7afa2810263653a34ba2c21c7a",
    "FEG": "0x389999216860ab8e0175387a0c90e5c52522c945",
    "PUSH": "0xf418588522d5dd018b425e472991e52ebbeeeeee",
    "HOP": "0xc5102fe9359fd9a28f877a67e36b0f050d81a3cc",
    "RSR": "0x320623b8e4ff03373931769a31fc52a4e78b5d70",
    "ORN": "0x0258f474786ddfd37abce6df6bbb1dd5dfc4434a",
    "SLP": "0x397ff1542f962076d0bfe58ea045ffa2d347aca0",
    "SUPER": "0xe53ec727dbdeb9e2d5456c3be40cff031ab40a55",
    "ALBT": "0x00a8b738e453ffd858a7edf03bccfe20412f0eb0",
    "sILV2": "0x7e77dcb127f99ece88230a64db8d595f31f1b068",
    "DODO": "0x43dfc4159d86f3a37a5a4b3d4580b888ad7d4ddd",
    "BOB": "0x7d8146cf21e8d7cbe46054e01588207b51198729",
    "GHST": "0x3f382dbd960e3a9bbceae22651e88158d2791550",
    "YFII": "0xa1d0e215a23d7030842fc67ce582a6afa3ccab83",
    "aDAI": "0x028171bca77440897b824ca71d1c56cac55b68a3",
    "DRGN": "0x419c4db4b9e25d6db2ad9691ccb832c8d9fda05e",
    "CELR": "0x4f9254c83eb525f9fcf346490bbb3ed28a81c667",
    "POOH": "0xb69753c06bb5c366be51e73bfc0cc2e3dc07e371",
    "TVK": "0xd084b83c305dafd76ae3e1b4e1f1fe2ecccb3988",
    "PICKLE": "0x429881672b9ae42b8eba0e26cd9c73711b891ca5",
    "LINA": "0x3e9bc21c9b189c09df3ef1b824798658d5011937",
    "cDAI": "0x5d3a536e4d6dbd6114cc1ead35777bab948e3643",
    "ANT": "0xa117000000f279d81a1d3cc75430faa017fa5a2e",
    "TNT": "0x08f5a9235b08173b7569f83645d2c7fb55e8ccd8",
    "WOOL": "0x8355dbe8b0e275abad27eb843f3eaf3fc855e525",
    "BITCOIN": "0x72e4f9f808c49a2a61de9c5896298920dc4eeea9",
    "LQTY": "0x6dea81c8171d0ba574754ef6f8b412f2ed88c54d",
    "AUDIO": "0x18aaa7115705e8be94bffebde57af9bfc265b998",
    "RLB": "0x046eee2cc3188071c02bfc1745a6b17c656e3f3d",
    "SWAP": "0xcc4304a31d09258b0029ea7fe63d032f52e44efe",
    "OHM": "0x64aa3364f17a4d01c6f1751fd97c2bd3d7e7f1d5",
    "RARI": "0xfca59cd816ab1ead66534d82bc21e7515ce441cf",
    "REP": "0xe94327d07fc17907b4db788e5adf2ed424addff6",
    "LADYS": "0x12970e6868f88f6557b76120662c1b3e50a646bf",
    "BTRFLY": "0xc55126051b22ebb829d00368f4b12bde432de5da",
    "bendWETH": "0xed1840223484483c0cb050e6fc344d1ebf0778a9",
    "TURBO": "0xa35923162c49cf95e6bf26623385eb431ad920d3",
    "0x0": "0x5a3e6a77ba2f983ec0d371ea3b475f8bc0811ad5",
    "PRE": "0xec213f83defb583af3a000b1c0ada660b1902a0f",
    "AIOZ": "0x626e8036deb333b408be468f951bdb42433cbf18",
    "crvUSD": "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e",
    "CRU": "0x32a7c02e79c4ea1008dd6564b35f131428673c41",
    "MIM": "0x99d8a9c45b2eca8864373a26d1459e3dff1e17f3",
    "SPONGE": "0x25722cd432d02895d9be45f5deb60fc479c8781e",
    "aUSDC": "0xbcca60bb61934080951369a648fb03df4f96263c",
    "KP3R": "0x1ceb5cb57c4d4e2b2433641b95dd330a33185a44",
    "STARL": "0x8e6cd950ad6ba651f6dd608dc70e5886b1aa6b24",
    "ShibDoge": "0x6adb2e268de2aa1abf6578e4a8119b960e02928f",
    "LUSD": "0x5f98805a4e8be255a32880fdec7f6728c6568ba0",
    "PSYOP": "0x3007083eaa95497cd6b2b809fb97b6a30bdf53d3",
    "steCRV": "0x06325440d014e39736583c165c2963ba99faf14e",
    "MAHA": "0xb4d930279552397bba2ee473229f89ec245bc365",
    "POLS": "0x83e6f1e41cdd28eaceb20cb649155049fac3d5aa",
    "RFD": "0x955d5c14c8d4944da1ea7836bd44d54a8ec35ba1",
    "INU": "0xc76d53f988820fe70e01eccb0248b312c2f1c7ca",
    "DPI": "0x1494ca1f11d487c2bbe4543e90080aeba4ba3c2b",
    "ARC": "0xc82e3db60a52cf7529253b4ec688f631aad9e7c2",
    "INJ": "0xe28b3b32b6c345a34ff64674606124dd5aceca30",
    "MIR": "0x09a3ecafa817268f77be1283176b946c4ff2e608",
    "DOGE2.0": "0xf2ec4a773ef90c58d98ea734c0ebdb538519b988",
    "MUTE": "0xa49d7499271ae71cd8ab9ac515e6694c755d400c",
    "EVERMOON": "0x4ad434b8cdc3aa5ac97932d6bd18b5d313ab0f6f",
    "DERC": "0x9fa69536d1cda4a04cfb50688294de75b505a9ae",
    "ADX": "0xade00c28244d5ce17d72e40330b1c318cd12b7c3",
    "HOICHI": "0xc4ee0aa2d993ca7c9263ecfa26c6f7e13009d2b6",
    "SDEX": "0x5de8ab7e27f6e7a1fff3e5b337584aa43961beef",
    "BAG": "0x235c8ee913d93c68d2902a8e0b5a643755705726",
    "XCN": "0xa2cd3d43c775978a96bdbf12d733d5a1ed94fb18",
    "ETH2x-FLI": "0xaa6e8127831c9de45ae56bb1b0d4d4da6e5665bd",
    "stkAAVE": "0x4da27a545c0c5b758a6ba100e3a049001de870f5",
    "GLM": "0x7dd9c5cba05e151c895fde1cf355c9a1d5da6429",
    "QOM": "0xa71d0588eaf47f12b13cf8ec750430d21df04974",
    "WOJAK": "0x5026f006b85729a8b14553fae6af249ad16c9aab",
    "DINO": "0x49642110b712c1fd7261bc074105e9e44676c68f",
    "Metis": "0x9e32b13ce7f2e80a01932b42553652e053d6ed8e",
    "REPv2": "0x221657776846890989a759ba2973e427dff5c9bb",
    "TRAC": "0xaa7a9ca87d3694b5755f213b5d04094b8d0f0a6f",
    "BEPRO": "0xcf3c8be2e2c42331da80ef210e9b1b307c03d36a",
}


def name_of(token: str) -> str:
    return f"mint-{token.lower()}"


def role_of(token: str) -> str:
    return "test" if token in TEST else "train"


def build(token: str, edges: pa.Table, labels: np.ndarray,
          provenance: dict[str, str]) -> TemporalGraph:
    if edges.column_names != COLUMNS:
        raise ValueError(f"{token}: unexpected edge list columns {edges.column_names}")
    if any(edges.column(c).null_count for c in COLUMNS):
        raise ValueError(f"{token}: the edge list has missing values")
    snapshot = edges.column("snapshot").to_numpy()
    T, E = int(snapshot[-1]), len(snapshot)
    step = np.diff(snapshot)
    if snapshot[0] != 1 or ((step != 0) & (step != 1)).any():
        raise ValueError(f"{token}: snapshots are not the contiguous run 1..T in order")
    if len(labels) != T or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{token}: {len(labels)} labels for {T} snapshots, or not 0/1")
    edge_ptr = np.concatenate([[0], np.cumsum(np.bincount(snapshot - 1, minlength=T))])

    days = pc.cast(edges.column("date"), pa.date32()).to_numpy().astype("datetime64[D]")
    days = days.astype(np.int64)
    first_day = int(days.min())
    day_counts = _day_counts(token, days - first_day, snapshot, T)

    both = pa.concat_arrays([edges.column("source").combine_chunks(),
                             edges.column("destination").combine_chunks()]).dictionary_encode()
    order = pc.sort_indices(both.dictionary).to_numpy()
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    ids = rank[both.indices.to_numpy()]
    addresses = both.dictionary.take(pa.array(order)).to_numpy(zero_copy_only=False)

    weight64 = edges.column("weight").to_numpy()
    weight = weight64.astype(np.float32)
    if not np.isfinite(weight).all() or ((weight == 0) & (weight64 != 0)).any():
        raise ValueError(f"{token}: weights do not fit float32")

    timestamps = (first_day + np.arange(T, dtype=np.int64)) * DAY
    role = role_of(token)
    g = TemporalGraph(
        name=name_of(token),
        domain="transaction",
        tasks=["graph_classification"],
        time_mode="discrete",
        timestamps=timestamps,
        edge_index=np.stack([ids[:E], ids[E:]]),
        edge_weight=weight,
        edge_ptr=edge_ptr.astype(np.int64),
        y={"edge_gs": Target(
            level="graph", kind="class", values=labels.astype(np.uint8), num_classes=2,
            source="labels_Edge_GS.csv: row i is snapshot i+1; 1 if the transfers of days "
                   "t+10 .. t+16 outnumber those of the snapshot's own days t .. t+6 (edge "
                   f"growth), else 0 ({CODE} script/utils/TGS.py, window_size=7, gap=3, "
                   "label_window_size=7)")},
        node_table={"address": addresses},
        splits=_splits(timestamps, role),
        meta={
            "freq": "1D",
            "time_note": "each snapshot is a 7-day window and the next starts one day later, "
                         "as released (the paper calls them weekly); timestamps are each "
                         "window's first day, UTC",
            "window": "7D",
            "timestamps_tz": "UTC, from the unix timestamps the release's dates derive from",
            "num_nodes": len(addresses),
            "channels": [],
            "covariate_channels": [],
            "directed": True,
            "multigraph": True,
            "edge_weight": {
                "kind": "token_amount",
                "observed": True,
                "units": "the ERC20 transfer value in the token's smallest unit",
                "description": "'weight' of the edge list, a float64 in the release, stored as "
                               "float32 (relative error at most 6e-8); zero and negative values "
                               "and self-loops are kept as released",
            },
            "adjacency": {"kind": "raw", "reference": f"{CODE} load_TGC_dataset builds each "
                          "snapshot as an unweighted undirected simple graph of source and "
                          "destination; the stored multigraph holds what it is built from"},
            "default_task": {
                "name": "graph_classification",
                "params": {"target": "edge_gs", "window": 1},
                "reference": f"{CODE} train_foundation_tgc_64.py: one graph label per "
                             "snapshot, predicted from that snapshot alone",
            },
            "day_counts": day_counts.tolist(),
            "day_counts_note": "transfers of each day from the first one; snapshot t holds "
                               "days t..t+6 in date order, so every row's date is recoverable",
            "first_date": str(np.datetime64(first_day, "D")),
            "role": role,
            "train_rank": TRAIN.index(token) + 1 if role == "train" else None,
            "token": token,
            "token_address": ADDRESS[token],
            "source": f"MiNT {token} transaction network (Shamsi, Ngo et al., 2025), Zenodo "
                      "record 15364297",
            "license": "cc-by-4.0",
            "license_note": "Zenodo lists CC BY 4.0 and MIT (the latter for the code)",
            "citation": CITATION,
            "provenance": provenance,
        },
    )
    validate(g)
    return g


def _day_counts(token: str, day: np.ndarray, snapshot: np.ndarray, T: int) -> np.ndarray:
    """Transfers per day, checking the release is the sliding window the format assumes."""
    offset = day - (snapshot - 1)
    if offset.min() < 0 or offset.max() >= WINDOW:
        raise ValueError(f"{token}: a transfer lies outside its snapshot's 7 days")
    if (np.diff(snapshot * WINDOW + offset) < 0).any():
        raise ValueError(f"{token}: transfers are not in date order within a snapshot")
    counts = np.bincount((snapshot - 1) * WINDOW + offset, minlength=T * WINDOW)
    counts = counts.reshape(T, WINDOW)
    day_counts = np.zeros(T + WINDOW - 1, dtype=np.int64)
    day_counts[: T] = counts[:, 0]
    day_counts[T:] = counts[-1, 1:]
    at = np.arange(T)[:, None] + np.arange(WINDOW)[None, :]
    if (day_counts[at] != counts).any():
        raise ValueError(f"{token}: snapshots disagree on the transfers of a day")
    return day_counts


def _splits(timestamps: np.ndarray, role: str) -> dict[str, Split]:
    """The code's cuts: val and test are floor(0.15 T) snapshots each, train is the rest."""
    T = len(timestamps)
    held = math.floor(T * 0.15)
    bounds = np.append(timestamps, timestamps[-1] + DAY)
    cut = [0, T - 2 * held, T - held, T]
    parts = {k: (int(bounds[cut[i]]), int(bounds[cut[i + 1]]))
             for i, k in enumerate(("train", "val", "test"))}
    splits = {"default": Split(
        boundaries=parts,
        reference=f"{CODE} train_foundation_tgc_64.py (test_ratio=val_ratio=0.15): the last "
                  "floor(0.15 T) snapshots are test, the floor(0.15 T) before them val and "
                  "the rest train; the paper reports 70/15/15")}
    if role == "test":
        splits["zero-shot"] = Split(
            boundaries={"test": (parts["val"][0], parts["test"][1])},
            reference=f"{CODE} test_foundation_tgc_64.py, which scores a held-out network on "
                      "its last 30% of snapshots (val and test of the default split)")
    return splits


def read_network(token: str, edgelists: zipfile.ZipFile, labels: zipfile.ZipFile,
                 provenance: dict[str, str]) -> tuple[pa.Table, np.ndarray, dict[str, str]]:
    member = next(n for n in edgelists.namelist() if n.endswith(f"/{token}_edgelist.txt"))
    types = {"source": pa.string(), "destination": pa.string(), "weight": pa.float64(),
             "date": pa.string(), "snapshot": pa.int64()}
    with edgelists.open(member) as f:
        edges = pcsv.read_csv(f, convert_options=pcsv.ConvertOptions(column_types=types))
    name = f"labels/{token}_labels.csv"
    column = pcsv.read_csv(io.BytesIO(labels.read(name)),
                           read_options=pcsv.ReadOptions(column_names=["label"]),
                           convert_options=pcsv.ConvertOptions(column_types={"label": pa.int64()}))
    provenance = {**provenance, f"{member} crc32": f"{edgelists.getinfo(member).CRC:08x}",
                  f"{name} crc32": f"{labels.getinfo(name).CRC:08x}"}
    return edges, column.column("label").to_numpy(), provenance


def open_release(edgelists: Path, labels: Path) -> tuple[zipfile.ZipFile, zipfile.ZipFile,
                                                         dict[str, str]]:
    outer = zipfile.ZipFile(labels)
    inner = zipfile.ZipFile(io.BytesIO(outer.read("TGS_labels_Edge_GS/TGS_labels.zip")))
    provenance = {"url": ZENODO, "MiNT_edgelists.zip sha256": _sha256(edgelists),
                  "MiNT_labels_Edge_GS.zip sha256": _sha256(labels)}
    return zipfile.ZipFile(edgelists), inner, provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edgelists", type=Path, required=True, help="MiNT_edgelists.zip")
    parser.add_argument("--labels", type=Path, required=True, help="MiNT_labels_Edge_GS.zip")
    parser.add_argument("--out", type=Path, required=True, help="one folder per network")
    parser.add_argument("--token", nargs="*", default=[*TRAIN, *TEST], choices=[*TRAIN, *TEST])
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    edgelists, labels, provenance = open_release(args.edgelists, args.labels)
    for token in args.token:
        g = build(token, *read_network(token, edgelists, labels, provenance))
        tgio.save(g, args.out / g.name)
        print(f"saved {g.name} ({g.num_steps} snapshots, {g.num_nodes} nodes)", flush=True)
        if args.push:
            print(hub.push(args.out / g.name, g.name), flush=True)


if __name__ == "__main__":
    main()
