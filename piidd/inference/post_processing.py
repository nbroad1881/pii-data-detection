import json
import argparse
import pickle
from pathlib import Path

import numpy as np
from datasets import Dataset
from scipy.special import softmax
from transformers import AutoConfig

from piidd.processing.post import (
    get_all_preds,
    check_name_casing,
    add_repeated_names,
    add_emails_and_urls,
    remove_name_titles,
    correct_name_student_preds,
    remove_bad_categories,
    check_phone_numbers,
)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/kaggle/input/pii-detection-removal-from-educational-data/test.json",
    )
    parser.add_argument("--model_dir", type=str, default="/kaggle/input/model-dir")
    parser.add_argument("--dataset_path", type=str, default="ds.pq")
    parser.add_argument("--tokenized_ds_path", type=str, default="tds.pq")
    parser.add_argument("--preds_path", type=str, default="preds.npy")
    parser.add_argument("--add_repeated_names", action="store_true")
    parser.add_argument("--add_emails_and_urls", action="store_true")
    parser.add_argument("--output_path", type=str, default="out")
    parser.add_argument("--output_csv_path", type=str, default="submission.csv")
    parser.add_argument("--include_token_text", action="store_true")
    parser.add_argument("--return_all_token_scores", action="store_true")
    parser.add_argument("--return_char_preds", action="store_true")
    parser.add_argument("--save_char_preds_path", type=str, default="char_preds.pkl")
    parser.add_argument("--thresholds", type=str, default="0.9")
    parser.add_argument("--remove_name_titles", action="store_true")
    parser.add_argument("--correct_name_student_preds", action="store_true")
    parser.add_argument("--remove_bad_categories", action="store_true")
    parser.add_argument("--check_phone_numbers", action="store_true")
    parser.add_argument("--calculate_f5", type=int, default=0)

    args = parser.parse_args()

    if args.calculate_f5:
        from piidd.training.utils import make_gt_dataframe, pii_fbeta_score_v2

    thresholds = [float(x) for x in args.thresholds.split(",")]

    ds = Dataset.from_parquet(args.dataset_path)
    tds = Dataset.from_parquet(args.tokenized_ds_path)

    predictions = softmax(np.load(args.preds_path), -1)

    try:
        config = json.load(open(Path(args.model_dir) / "config.json"))
    except FileNotFoundError:
        config = AutoConfig.from_pretrained(args.model_dir).to_dict()

    id2label = {int(i): ll for i, ll in config["id2label"].items()}

    output_dir = Path(args.output_path)

    output_dir.mkdir(exist_ok=True, parents=True)


    data = json.load(open(args.data_path))

    if args.return_all_token_scores:
        pred_df = get_all_preds(
            predictions,
            ds,
            tds,
            id2label,
            return_char_preds=False,
            return_all_token_scores=True,
        )
        pred_df.to_parquet(args.output_csv_path, index=False)
        return

    if args.calculate_f5:
        gt_df = make_gt_dataframe(ds)

    f5s = []
    for th in thresholds:
        results = get_all_preds(
            predictions,
            ds,
            tds,
            id2label,
            threshold=th,
            return_char_preds=args.return_char_preds,
        )

        if args.return_char_preds:
            pred_df, char_preds = results
        else:
            pred_df = results

        pred_df = check_name_casing(pred_df)

        if args.calculate_f5:
            f5 = pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5)

            f5s.append((f"f5-threshold={round(th, 2)}", round(f5, 5)))

        if args.add_repeated_names:
            pred_df = add_repeated_names(pred_df, data)

            if args.calculate_f5:
                f5 = pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5)

                f5s.append((f"f5-threshold={round(th, 2)}-repeated_names", round(f5, 5)))

        if args.add_emails_and_urls:
            pred_df = add_emails_and_urls(pred_df, data)

            if args.calculate_f5:
                f5 = pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5)

                f5s.append((f"f5-threshold={round(th, 2)}-emails_urls", round(f5, 5)))

        if args.remove_name_titles:
            pred_df = remove_name_titles(pred_df)

            if args.calculate_f5:
                f5 = pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5)

                f5s.append(
                    (f"f5-threshold={round(th, 2)}-remove_name_titles", round(f5, 5))
                )

        if args.correct_name_student_preds:
            pred_df = correct_name_student_preds(pred_df)

            if args.calculate_f5:
                f5 = pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5)

                f5s.append(
                    (
                        f"f5-threshold={round(th, 2)}-correct_name_student_preds",
                        round(f5, 5),
                    )
                )

        if args.remove_bad_categories:
            pred_df = remove_bad_categories(pred_df)

            if args.calculate_f5:
                f5 = pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5)

                f5s.append(
                    (f"f5-threshold={round(th, 2)}-remove_bad_categories", round(f5, 5))
                )

        if args.check_phone_numbers:
            pred_df = check_phone_numbers(pred_df)


            if args.calculate_f5:
                f5 = pii_fbeta_score_v2(pred_df, gt_df, output_dir, beta=5)

                f5s.append(
                    (f"f5-threshold={round(th, 2)}-check_phone_numbers", round(f5, 5))
                )

    print(f5s)

    if args.return_char_preds:
        with open(args.save_char_preds_path, "wb") as f:
            pickle.dump(char_preds, f)

    pred_df["row_id"] = range(len(pred_df))

    cols2save = ["row_id", "document", "token", "label"]

    if args.include_token_text:
        cols2save.append("token_text")

    pred_df[cols2save].to_csv(args.output_csv_path, index=False)


if __name__ == "__main__":
    main()
