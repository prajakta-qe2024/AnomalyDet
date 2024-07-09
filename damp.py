import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from damp_utils import contains_constant_regions, nextpow2, MASS_V2

def DAMP_2_0(
    time_series: np.ndarray,
    subsequence_length: int,
    stride: int,
    location_to_start_processing: int,
    lookahead: int = 0,
    enable_output: bool = True,
    data_set_name = '_'
) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """Computes DAMP of a time series and detects anomalies.
    
    Args:
        time_series (np.ndarray): Univariate time series
        subsequence_length (int): Window size
        stride (int): Window stride
        location_to_start_processing (int): Start/End index of test/train set
        lookahead (int, optional): How far to look ahead for pruning. Defaults to 0.
        enable_output (bool, optional): Print results and save plot. Defaults to True.

    Returns:
        Tuple[np.ndarray, float, int, np.ndarray]: Matrix profile, discord score, its corresponding position in the profile, and anomaly indices
    """
    assert (subsequence_length > 10) and (
        subsequence_length <= 1000
    ), "`subsequence_length` must be > 10 or <= 1000."

    if lookahead is None:
        lookahead = int(2 ** nextpow2(16 * subsequence_length))
    elif (lookahead != 0) and (lookahead != 2 ** nextpow2(lookahead)):
        lookahead = int(2 ** nextpow2(lookahead))

    if contains_constant_regions(
        time_series=time_series, subsequence_length=subsequence_length
    ):
        raise Exception(
            "ERROR: This dataset contains constant and/or near constant regions.\nWe define the time series with an overall variance less than 0.2 or with a constant region within its sliding window as the time series containing constant and/or near constant regions.\nSuch regions can cause both false positives and false negatives depending on how you define anomalies.\nAnd more importantly, it can also result in imaginary numbers in the calculated Left Matrix Profile, from which we cannot get the correct score value and position of the top discord.\n** The program has been terminated. **"
        )

    if (location_to_start_processing / subsequence_length) < 4:
        print(
            "WARNING: location_to_start_processing/subsequence_length is less than four.\nWe recommend that you allow DAMP to see at least four cycles, otherwise you may get false positives early on.\nIf you have training data from the same domain, you can prepend the training data, like this Data = [trainingdata, testdata], and call DAMP(data, S, length(trainingdata))"
        )
        if location_to_start_processing < subsequence_length:
            print(
                f"location_to_start_processing cannot be less than the subsequence length.\nlocation_to_start_processing has been set to {subsequence_length}"
            )
            location_to_start_processing = subsequence_length
        print("------------------------------------------\n\n")
    else:
        if location_to_start_processing > (len(time_series) - subsequence_length + 1):
            print(
                "WARNING: location_to_start_processing cannot be greater than length(time_series)-S+1"
            )
            location_to_start_processing = len(time_series) - subsequence_length + 1
            print(
                f"location_to_start_processing has been set to {location_to_start_processing}"
            )
            print("------------------------------------------\n\n")

    left_mp = np.zeros(time_series.shape)
    best_so_far = -np.inf
    bool_vec = np.ones(len(time_series))

    for i in range(
        location_to_start_processing - 1,
        location_to_start_processing + 16 * subsequence_length,
        stride,
    ):
        if not bool_vec[i]:
            left_mp[i] = left_mp[i - 1] - 1e-05
            continue

        if i + subsequence_length - 1 > len(time_series):
            break

        query = time_series[i : i + subsequence_length]
        left_mp[i] = np.amin(MASS_V2(time_series[:i], query))
        best_so_far = np.amax(left_mp)

        if lookahead != 0:
            start_of_mass = min(i + subsequence_length - 1, len(time_series))
            end_of_mass = min(start_of_mass + lookahead - 1, len(time_series))

            if (end_of_mass - start_of_mass + 1) > subsequence_length:
                distance_profile = MASS_V2(
                    time_series[start_of_mass : end_of_mass + 1], query
                )

                dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]
                ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass
                bool_vec[ts_index_less_than_BSF] = 0

    for i in range(
        location_to_start_processing + 16 * subsequence_length,
        len(time_series) - subsequence_length + 1,
        stride,
    ):
        if not bool_vec[i]:
            left_mp[i] = left_mp[i - 1] - 1e-05
            continue

        approximate_distance = np.inf
        X = int(2 ** nextpow2(8 * subsequence_length))
        flag = 1
        expansion_num = 0

        if i + subsequence_length - 1 > len(time_series):
            break

        query = time_series[i : i + subsequence_length]

        while approximate_distance >= best_so_far:
            if i - X + 1 + (expansion_num * subsequence_length) < 1:
                approximate_distance = np.amin(MASS_V2(x=time_series[: i + 1], y=query))
                left_mp[i] = approximate_distance

                if approximate_distance > best_so_far:
                    best_so_far = approximate_distance

                break

            if flag == 1:
                flag = 0
                approximate_distance = np.amin(
                    MASS_V2(time_series[i - X + 1 : i + 1], query)
                )
            else:
                X_start = i - X + 1 + (expansion_num * subsequence_length)
                X_end = i - X // 2 + (expansion_num * subsequence_length)
                approximate_distance = np.amin(
                    MASS_V2(x=time_series[X_start : X_end + 1], y=query)
                )

            if approximate_distance < best_so_far:
                left_mp[i] = approximate_distance
                break

            X *= 2
            expansion_num += 1

        if lookahead != 0:
            start_of_mass = min(i + subsequence_length, len(time_series))
            end_of_mass = min(start_of_mass + lookahead - 1, len(time_series))

            if (end_of_mass - start_of_mass) > subsequence_length:
                distance_profile = MASS_V2(
                    x=time_series[start_of_mass : end_of_mass + 1], y=query
                )

                dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]
                ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass
                bool_vec[ts_index_less_than_BSF] = 0

    PV = bool_vec[
        location_to_start_processing - 1 : len(time_series) - subsequence_length + 1
    ]
    PR = (len(PV) - sum(PV)) / len(PV)

    discord_score, position = np.amax(left_mp), np.argmax(left_mp)
    print("\nResults:")
    print(f"Pruning Rate: {PR}")
    print(f"Predicted discord score/position: {discord_score} / {position}")

    anomalies = np.where(left_mp > np.percentile(left_mp, 95))[0]

    if enable_output:
        save_name = f"{data_set_name}_damp_{subsequence_length}_{stride}_{location_to_start_processing}_{lookahead}"

        abs_left_mp = abs(left_mp)
        os.makedirs("./outputs", exist_ok=True)
        damp_save_path = f"./outputs/{save_name}.npy"
        np.save(damp_save_path, left_mp)

        plt.figure(figsize=(30, 20))

        plt.subplot(2, 1, 1)
        plt.plot(
            (time_series - np.min(time_series))
            / (np.max(time_series) - np.min(time_series))
            + 1.1,
            c="black",
            label="Time Series"
        )
        plt.scatter(anomalies, [(time_series[i] - np.min(time_series)) / (np.max(time_series) - np.min(time_series)) + 1.1 for i in anomalies], c='red', label='Anomalies')
        plt.title("Time Series with Anomalies", fontdict={"fontsize": 40})
        plt.xlabel("Index", fontdict={"fontsize": 20})
        plt.ylabel("Normalized Value", fontdict={"fontsize": 20})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(abs_left_mp / np.max(abs_left_mp), c="blue", label="DAMP")
        plt.scatter(anomalies, [left_mp[i] / np.max(abs_left_mp) for i in anomalies], c='red', label='Anomalies')
        plt.title("DAMP with Anomalies", fontdict={"fontsize": 40})
        plt.xlabel("Index", fontdict={"fontsize": 20})
        plt.ylabel("Value", fontdict={"fontsize": 20})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()

        os.makedirs("./figures", exist_ok=True)
        plot_save_path = f"./figures/{save_name}.png"

        plt.tight_layout()
        plt.savefig(plot_save_path)
        print(f"Saved figure in {plot_save_path}")

    return left_mp, discord_score, position, anomalies
import pandas as pd
if __name__ == "__main__":
    # Set parameters
    parser = argparse.ArgumentParser(description="Set parameters")
    parser.add_argument("--subsequence_length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--location_to_start_processing", type=int, default=31)
    parser.add_argument("--lookahead", type=int, default=None)
    parser.add_argument("--enable_output", action="store_true")
    args = parser.parse_args()

    # Load data
    yahoo_data_path = "/Users/prajaktadarade/Downloads/all_stocks_5yr.csv"
    grouped = pd.read_csv(yahoo_data_path).groupby('Name')
    first_group_key = list(grouped.groups.keys())[3]
    time_series = grouped.get_group(first_group_key)['open'].values
    # data_path = "/Users/prajaktadarade/Documents/Quadeye/ydata-labeled-time-series-anomalies-v1_0 2/A1Benchmark/"
    # time_series = pd.read_csv(f"{data_path}/real_1.csv")['value'].values
    # time_series = np.loadtxt("data/samples/BourkeStreetMall.txt")
    for k, df in grouped:
    # Run DAMP  
        for col in ['open', 'volume']:
            time_series = df[col].values
            DAMP_2_0(
                time_series=time_series,
                subsequence_length=args.subsequence_length,
                stride=args.stride,
                location_to_start_processing=args.location_to_start_processing,
                lookahead=args.lookahead,
                enable_output=args.enable_output,
                data_set_name = f'{k}_{col}'
            )