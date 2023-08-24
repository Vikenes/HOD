from pathlib import Path
import pandas as pd

default_emulator_data_path = Path("/cosma6/data/dp004/dc-cues1/emulator/")


def snapshot_to_scale_factor(
    snapshot: int, emulator_data_path: str = default_emulator_data_path
) -> float:
    return pd.read_csv(emulator_data_path / "scales.dat", header=None).iloc[snapshot, 0]


def snapshot_to_redshift(
    snapshot: int, emulator_data_path: str = default_emulator_data_path
) -> float:
    scale_factor = snapshot_to_scale_factor(
        snapshot=snapshot, emulator_data_path=emulator_data_path
    )
    return 1.0 / scale_factor - 1.0


if __name__ == '__main__':
    for snapNum in range(21):
        redshift = snapshot_to_redshift(snapshot=snapNum,)
        print(f"snapshot {snapNum} - redshift{redshift:.3f}")