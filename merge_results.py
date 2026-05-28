import os
import json
import argparse

from postprocess import extract_and_validate_hoi_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge prediction JSON files in a folder")
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing prediction JSON files')
    parser.add_argument('--re', action='store_true', help='Re-run postprocessing on the raw model outputs instead of reusing the stored preds')
    args = parser.parse_args()

    folder_path = args.folder_path
    repostprocess = args.re

    # Collect all entries (each entry pairs a pred with its gt_path)
    all_entries = []

    # Recursively walk the folder and load every JSON file
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not filename.endswith('.json'):
                continue
            if "merged_output" in filename:
                continue
            file_path = os.path.join(root, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Ensure the expected fields are present
            if 'preds' not in data or 'gt_paths' not in data:
                print(f"Skipping {file_path}: missing 'preds' or 'gt_paths' fields")
                continue

            # preds/gt_paths (and outputs/gt_paths) are assumed to be aligned one-to-one
            if repostprocess:
                for out_item, gt_item in zip(data['outputs'], data['gt_paths']):
                    pred = extract_and_validate_hoi_output(out_item)
                    all_entries.append({
                        'pred': pred,
                        'gt_path': gt_item
                    })
            else:
                for pred_item, gt_item in zip(data['preds'], data['gt_paths']):
                    all_entries.append({
                        'pred': pred_item,
                        'gt_path': gt_item
                    })

    # Sort by the file_name field in gt_path
    all_entries.sort(key=lambda x: x['gt_path']['file_name'])

    # Filter out duplicate images
    all_entries_filtered = []
    seen_gt_paths = set()
    for entry in all_entries:
        gt_path = entry['gt_path']['file_name']
        if gt_path not in seen_gt_paths:
            seen_gt_paths.add(gt_path)
            all_entries_filtered.append(entry)
        else:
            print(f"Duplicate entry found and skipped: {gt_path}")

    # Split the sorted, de-duplicated entries back into preds and gt_paths
    merged_preds = [entry['pred'] for entry in all_entries_filtered]
    merged_gt_paths = [entry['gt_path'] for entry in all_entries_filtered]

    print(f'Total entries after filtering: {len(merged_preds)}')

    merged_data = {
        'preds': merged_preds,
        'gt_paths': merged_gt_paths
    }

    output_path = os.path.join(folder_path, 'merged_output.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f'Merged JSON saved to: {output_path}')
