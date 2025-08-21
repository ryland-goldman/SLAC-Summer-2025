"""
ModeratorProcess2.py

Generates .dat files

"""

import math
import numpy as np
import pandas as pd

data_dir = "data-dir"

# -----------------------------
# Physics helpers
# -----------------------------
def momentum_to_ke(p_mev_c: np.ndarray | float) -> np.ndarray | float:
    """Convert momentum (MeV/c) to kinetic energy (keV) for electrons.
    Uses electron rest mass m_e = 0.510998950 MeV.
    Works with scalars or NumPy arrays.
    """
    m_e = 0.510998950  # MeV
    total_energy = np.sqrt(np.asarray(p_mev_c) ** 2 + m_e ** 2)
    ke_keV = (total_energy - m_e) * 1e3
    return ke_keV


# -----------------------------
# Core processing
# -----------------------------
def run_sum(threadnumber: str) -> None:
    """Aggregate moderator outputs for a given thread number.

    Input files (whitespace-delimited, with a comment header line):
      - GModeratorOutC{thread}.txt : initial state
      - GModeratorOut{thread}.txt  : stop state
      - GModeratorOutB{thread}.txt : newly created particles

    Output parquet files (Brotli-compressed):
      - Out{thread}.dat    : normal tracks (with end position/time)
      - Out_r{thread}.dat  : reflection/ambiguous initial tracks
      - Out_a{thread}.dat  : annihilation-classified tracks
    """

    # Common schema for fast parsing
    parse_kwargs = dict(
        skiprows=1,
        sep=r"\s+",
        dtype={
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
            "Px": np.float32,
            "Py": np.float32,
            "Pz": np.float32,
            "t": np.float32,
            "PDGid": str,
            "EventID": np.uint32,
            "TrackID": np.uint16,
        },
        usecols=["x", "y", "z", "Px", "Py", "Pz", "t", "PDGid", "EventID", "TrackID"],
        on_bad_lines="skip",
        names="x y z Px Py Pz t PDGid EventID TrackID ParentID Weight".split(" "),
        comment="#",
        engine="c",
        memory_map=True,
    )

    try:
        # -------------------------
        # Read & filter once
        # -------------------------
        df_initial = pd.read_csv(f"{data_dir}/ModeratorOutC{threadnumber}.txt", **parse_kwargs)
        df_stop = pd.read_csv(f"{data_dir}/ModeratorOut{threadnumber}.txt", **parse_kwargs)
        df_new = pd.read_csv(f"{data_dir}/ModeratorOutB{threadnumber}.txt", **parse_kwargs)

        # Filter by PDG ID and drop the column early to save memory
        df_initial = df_initial[df_initial["PDGid"] == "-11"].drop(columns=["PDGid"])  # e+
        df_stop = df_stop[df_stop["PDGid"] == "-11"].drop(columns=["PDGid"])          # e+
        df_new = df_new[df_new["PDGid"] == "22"].drop(columns=["PDGid"])               # gamma

        # -------------------------
        # Derive per-(EventID, TrackID) initial features
        # -------------------------
        # Keep the first row per (EventID, TrackID) and count rows to flag reflections
        init_grouped = (
            df_initial
            .groupby(["EventID", "TrackID"], as_index=False)
            .agg({"x": "first", "y": "first", "z": "first", "Px": "first", "Py": "first", "Pz": "first", "t": "first", "EventID": "first", "TrackID": "first"})
        )
        init_counts = df_initial.groupby(["EventID", "TrackID"], as_index=False).size().rename(columns={"size": "init_count"})
        init = init_grouped.merge(init_counts, on=["EventID", "TrackID"], how="left")

        # Compute momentum magnitude, KE, and polar angle (deg)
        initial_p = np.sqrt(init["Px"] ** 2 + init["Py"] ** 2 + init["Pz"] ** 2)
        initial_e = momentum_to_ke(initial_p)
        # Avoid invalid values if initial_p is 0 by clipping the ratio to [-1, 1]
        cos_theta = np.divide(init["Pz"], initial_p, out=np.zeros_like(initial_p, dtype=np.float32), where=initial_p != 0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        initial_angle = np.degrees(np.arccos(cos_theta)).astype(np.float32)

        init = init.assign(
            initialx=init["x"].astype(np.float32),
            initialy=init["y"].astype(np.float32),
            initialz=init["z"].astype(np.float32),
            initialPx=init["Px"].astype(np.float32),
            initialPy=init["Py"].astype(np.float32),
            initialPz=init["Pz"].astype(np.float32),
            initialP=initial_p.astype(np.float32),
            initialE=initial_e.astype(np.float32),
            initialAngle=initial_angle,
        )

        # Reflection/ambiguous: groups where we did not get exactly one initial row
        reflect_mask = init["init_count"] != 1
        reflect_df = (
            init.loc[reflect_mask, [
                "initialx", "initialy", "initialz",
                "initialPx", "initialPy", "initialPz",
                "initialP", "initialE", "initialAngle",
                "EventID", "TrackID"
            ]]
            .assign(RunID=0)
        )

        # -------------------------
        # Derive per-(EventID, TrackID) stop (end) features
        # -------------------------
        stop = (
            df_stop
            .groupby(["EventID", "TrackID"], as_index=False)
            .agg({"x": "first", "y": "first", "z": "first", "t": "first"})
            .rename(columns={"x": "endx", "y": "endy", "z": "endz", "t": "endt"})
        )

        # Keep only tracks with exactly one initial row for main/anni classification
        init_valid = init.loc[~reflect_mask, [
            "EventID", "TrackID",
            "initialx", "initialy", "initialz",
            "initialPx", "initialPy", "initialPz",
            "initialP", "initialE", "initialAngle"
        ]]

        # Join initial->stop; tracks missing a stop will drop out (matches original behavior of using first row if present)
        tracks = init_valid.merge(stop, on=["EventID", "TrackID"], how="inner")

        # -------------------------
        # Vectorized annihilation classification
        # For each event, compute min r2 to any gamma (22) at the same event
        # r2 = dx^2 + dy^2 + dz^2 + dt^2; anni if min_r2 ~ 0 within 1e-3
        # -------------------------
        if not df_new.empty and not tracks.empty:
            # Cross-join within each EventID by merging on EventID
            new_pos = df_new[["EventID", "x", "y", "z", "t"]].rename(columns={"x": "gx", "y": "gy", "z": "gz", "t": "gt"})
            cross = tracks.merge(new_pos, on="EventID", how="left")

            # Compute squared distance in (x,y,z,t)
            dx = cross["endx"] - cross["gx"]
            dy = cross["endy"] - cross["gy"]
            dz = cross["endz"] - cross["gz"]
            dt = cross["endt"] - cross["gt"]
            r2 = dx * dx + dy * dy + dz * dz + dt * dt
            cross = cross.assign(r2=r2)

            # Reduce to per-track min r2
            min_r2 = (
                cross.groupby(["EventID", "TrackID"], as_index=False)["r2"].min()
                .rename(columns={"r2": "min_r2"})
            )
            tracks = tracks.merge(min_r2, on=["EventID", "TrackID"], how="left")
        else:
            # No new gammas to compare; mark as non-annihilation by default
            tracks = tracks.assign(min_r2=np.inf)

        anni_tol = 1e-3
        is_anni = tracks["min_r2"].fillna(np.inf) <= anni_tol

        # -------------------------
        # Build output DataFrames with consistent dtypes
        # -------------------------
        common_cols = [
            "initialx", "initialy", "initialz",
            "initialPx", "initialPy", "initialPz",
            "initialP", "initialE", "initialAngle",
        ]

        main_cols = common_cols + ["endx", "endy", "endz", "endt", "EventID", "TrackID", "RunID"]
        anni_cols = main_cols
        reflect_cols = common_cols + ["EventID", "TrackID", "RunID"]

        df = (
            tracks.loc[~is_anni, common_cols + ["endx", "endy", "endz", "endt", "EventID", "TrackID"]]
            .assign(RunID=0)
        )
        anni_df = (
            tracks.loc[is_anni, common_cols + ["endx", "endy", "endz", "endt", "EventID", "TrackID"]]
            .assign(RunID=0)
        )

        # Combine reflection df prepared earlier
        # Ensure dtype consistency
        for c in common_cols:
            reflect_df[c] = reflect_df[c].astype("float32")
        for c in common_cols + ["endx", "endy", "endz", "endt"]:
            if c in df.columns:
                df[c] = df[c].astype("float32")
            if c in anni_df.columns:
                anni_df[c] = anni_df[c].astype("float32")

        for c in ["EventID", "TrackID"]:
            if c in df.columns:
                df[c] = df[c].astype("Int32")
            if c in anni_df.columns:
                anni_df[c] = anni_df[c].astype("Int32")
            reflect_df[c] = reflect_df[c].astype("Int32")

        # RunID is a scalar 0
        df["RunID"] = df.get("RunID", 0).astype("Int32")
        reflect_df["RunID"] = reflect_df.get("RunID", 0).astype("Int32")
        anni_df["RunID"] = anni_df.get("RunID", 0).astype("Int32")

        # Reorder columns explicitly
        df = df[main_cols]
        anni_df = anni_df[anni_cols]
        reflect_df = reflect_df[reflect_cols]

        # -------------------------
        # Persist outputs
        # -------------------------
        df.to_parquet(f"{data_dir}/Out{threadnumber}.dat", engine="pyarrow", compression="brotli", compression_level=10, index=False)
        reflect_df.to_parquet(f"{data_dir}/Out_r{threadnumber}.dat", engine="pyarrow", compression="brotli", compression_level=10, index=False)
        anni_df.to_parquet(f"{data_dir}/Out_a{threadnumber}.dat", engine="pyarrow", compression="brotli", compression_level=10, index=False)

    except Exception as e:
        # Stay consistent with the original log format while avoiding the undefined variable 'i'
        print(f"Thread {threadnumber} failed with exception: {e}")


if __name__ == "__main__":
    import sys
    run_sum(sys.argv[1])