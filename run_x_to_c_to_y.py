import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import argparse
import os
import gc
import torch

from src.models.Explicd import Explicd
from src.models.MMed_Llama_3_8B import MMedLlama3
from src.models.Mistral import Mistral
from src.models.GPT4o import GPT4o
from src.utils import map_label_to_name, load_data, map_letter_to_label, calculate_metrics, save_data_to_json, seed_everything, get_current_date, create_explicd_config
from src.rices import RICES


def x_to_c(dataset: str, split: int=None, raw_values: bool=False, predict_for_train_set: bool=False) -> None:
    """Predicts concepts from ExpLICD model with self-refine.

    Args:
        dataset (str): Name of the dataset.
        split (int, optional): Split to use in the case of PH2. Defaults to None.
        raw_values (bool, optional): Whether to use raw concepts or not. Defaults to False.
        predict_for_train_set (bool, optional): Whether to generate the reports for training set.

    Returns:
        None: Save predicted concepts into a CSV file.
    """

    # Load data
    train_dataloader, test_dataloader = load_data(
        dataset=dataset, 
        split=split,
        data_path=getattr(args, 'data_path', None)
    )

    # Initialize ExpLICD model
    config = create_explicd_config(gpu_id=0)
    model = Explicd(config=config)

    # ==================== TEST SET PROCESSING ====================
    dict_to_save_data = dict()
    print("\n[INFO] Processing TEST set...")
    
    for batch in tqdm(test_dataloader, desc="Test set"):
        img_ids = batch["img_id"]
        y_true = batch["class_label"].numpy()

        # ============ EXPLICD SELF-REFINE INTEGRATION ============
        # Toggle use_self_refine to compare baseline vs refined predictions
        use_self_refine = True  # Set to False for baseline (no refinement)
        
        predicted_concepts, raw_scores, refinement_info = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=use_self_refine
        )
        
        # Log refinement statistics (only when self-refine is enabled)
        if use_self_refine and refinement_info is not None:
            print(f"[TEST] {batch['img_id'][0]}: "
                  f"{refinement_info['initial_violations']} → "
                  f"{refinement_info['final_violations']} violations "
                  f"({refinement_info['iterations']} iterations)")
        # ========================================================

        # Assemble report template
        if not raw_values:
            # Add diagnosis to refined concepts
            report_template = predicted_concepts + f" Thus the diagnosis is {map_label_to_name(y_true)}."
        else:
            # Save raw scores if requested
            report_template = {
                "raw_scores": raw_scores,
                "concepts": predicted_concepts
            }

        dict_to_save_data[img_ids[0]] = str(report_template)
    
    # ==================== TRAIN SET PROCESSING (if requested) ====================
    if predict_for_train_set:
        print("\n[INFO] Processing TRAIN set...")
        
        for batch in tqdm(train_dataloader, desc="Train set"):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()

            # ============ EXPLICD SELF-REFINE INTEGRATION ============
            # Toggle use_self_refine to compare baseline vs refined predictions
            use_self_refine = True  # Set to False for baseline (no refinement)
            
            predicted_concepts, raw_scores, refinement_info = model.get_concept_predictions_with_self_refine(
                batch=batch, 
                config=config,
                use_self_refine=use_self_refine
            )
            
            # Log refinement statistics (only when self-refine is enabled)
            if use_self_refine and refinement_info is not None:
                print(f"[TRAIN] {batch['img_id'][0]}: "
                      f"{refinement_info['initial_violations']} → "
                      f"{refinement_info['final_violations']} violations "
                      f"({refinement_info['iterations']} iterations)")
            # ========================================================

            # Assemble report template
            if not raw_values:
                # Add diagnosis to refined concepts
                report_template = predicted_concepts + f" Thus the diagnosis is {map_label_to_name(y_true)}."
            else:
                # Save raw scores if requested
                report_template = {
                    "raw_scores": raw_scores,
                    "concepts": predicted_concepts
                }

            dict_to_save_data[img_ids[0]] = str(report_template)

    # Save reports into CSV file
    pre_df = pd.DataFrame.from_dict(dict_to_save_data, orient='index', columns=['Column 2']).reset_index()
    pre_df.columns = ["image_id", "report"]
    
    if split is None:
        file_path = f"results/concept_prediction/{dataset}_dermatology_reports_generated_by_Explicd_raw_values_{raw_values}.csv"
    else:
        file_path = f"results/concept_prediction/{dataset}_split_{split}_dermatology_reports_generated_by_Explicd_raw_values_{raw_values}.csv"

    # Extract the directory path from the file path
    dir_path = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    pre_df.to_csv(file_path, index=False)
    print(f"\n[SUCCESS] Saved concept predictions to {file_path}")

    # Free GPU memory
    del model
    del test_dataloader
    del dict_to_save_data
    gc.collect()
    torch.cuda.empty_cache()


def c_to_y(model_name: str, dataset:str, ckpt:str, split=None, raw_values=False, report_path: str = None, use_demos=False, n_demos=0, ground_truth_concepts=False):
    """
    Predict final diagnosis from concepts using LLM.
    
    Report template:
    > The color is..., the shape is..., ... Thus the diagnosis is {label}.
    """

    # Load reports
    if dataset == 'PH2':
        if report_path is not None:
            df_reports = pd.read_csv(report_path) 
        else:  
            df_reports = pd.read_csv(f"results/concept_prediction/PH2_split_{split}_dermatology_reports_generated_by_Explicd_raw_values_{raw_values}.csv")
        
        PH2_TEST = pd.read_csv(f"data/PH2/splits/PH2_test_split_{split}.csv")
        PH2_TRAIN = pd.read_csv(f"data/PH2/splits/PH2_train_split_{split}.csv")
        df_reports_test = df_reports.loc[df_reports.image_id.isin(PH2_TEST.images.to_list())]
        df_reports_train = df_reports.loc[df_reports.image_id.isin(PH2_TRAIN.images.to_list())]
        
    elif dataset == 'Derm7pt':
        if report_path is not None:
            df_reports = pd.read_csv(report_path) 
        else:
            df_reports = pd.read_csv(f"results/concept_prediction/Derm7pt_dermatology_reports_generated_by_Explicd_raw_values_{raw_values}.csv")
        
        D7_TEST = pd.read_csv("data/Derm7pt/splits/derm7pt_test.csv")
        D7_TRAIN = pd.read_csv("data/Derm7pt/splits/derm7pt_train.csv")
        df_reports_test = df_reports.loc[df_reports.image_id.isin(D7_TEST.images.to_list())]
        df_reports_train = df_reports.loc[df_reports.image_id.isin(D7_TRAIN.images.to_list())]
        
    elif dataset == 'HAM10000':
        if report_path is not None:
            df_reports = pd.read_csv(report_path) 
        else:
            df_reports = pd.read_csv(f"results/concept_prediction/HAM10000_dermatology_reports_generated_by_Explicd_raw_values_{raw_values}.csv")

        HAM_TEST = pd.read_csv("data/HAM10000/splits/HAM10000_test.csv")
        HAM_TRAIN = pd.read_csv("data/HAM10000/splits/HAM10000_train.csv")
        HAM_VAL = pd.read_csv("data/HAM10000/splits/HAM10000_val.csv")
        df_reports_test = df_reports.loc[df_reports.image_id.isin(HAM_TEST.image_id.to_list())]
        df_reports_train = pd.concat([df_reports.loc[df_reports.image_id.isin(HAM_TRAIN.image_id.to_list())], 
                                      df_reports.loc[df_reports.image_id.isin(HAM_VAL.image_id.to_list())]])
    else:
        raise ValueError(f"The dataset {dataset} is not implemented.")

    # Initialize LLM
    if model_name == "MMed":
        model = MMedLlama3(ckpt)
    elif model_name == "Mistral":
        model = Mistral()
    elif model_name == "GPT":
        model = GPT4o(model=ckpt)
    else:
        raise TypeError(f"The specified model {model_name} does not have a valid implementation.")

    dict_responses = {
        'image_id': [],
        'gt_response': [],
        'llm_response': [],
        'demonstrations': [],
        'predicted_concepts': []
    }

    # Define instruction and query
    instruction = f"You're a english doctor, make a good choice based on the question and options. You need to answer the letter of the option without further explanations."
    query = """###Question: What is the type of skin lesion that is associated with the following dermoscopic concepts: {}. ###Options: A. Nevus\nB. Melanoma. ###Answer:"""

    # Demonstrations
    if use_demos:
        rices = RICES(dataset=dataset, split=split, feature_extractor="explicd", valid_ids=[])

    for img_id, report in tqdm(zip(df_reports_test.image_id.to_list(), df_reports_test.report.to_list()), desc="Predicting labels"):

        # Demonstrations
        if use_demos:
            # Get most similar N image ids to the query image
            demos_ids = rices.get_context_keys(key=img_id, n=n_demos)
            demos_to_use_in_prompt = []
            # Iterate over retrieved demo_ids and save the respective report into a list
            for id in demos_ids:
                sample = df_reports_train[df_reports_train.image_id == id].report.to_list()
                demos_to_use_in_prompt.append(sample[0])
        else:
            demos_to_use_in_prompt = None
    
        # Extract concepts from ExpLICD report format
        concepts = report[:report.find("Thus the diagnosis is")-1]
        input_query = query.format(concepts)
        gt_response = report[report.find("Thus the diagnosis is ")+len("Thus the diagnosis is "):-1]
        
        if model_name == "GPT":
            llm_response = map_letter_to_label(model.inference_text(instruction=instruction, query=input_query, max_new_tokens=1).strip())
        else:
            prompt = model.get_prompt(instruction, input_query, demos=demos_to_use_in_prompt)
            llm_response = map_letter_to_label(model.predict(prompt, max_new_tokens=1).strip())
            
        dict_responses['image_id'].append(img_id)
        dict_responses['gt_response'].append(gt_response)
        dict_responses['llm_response'].append(llm_response)
        dict_responses['demonstrations'].append(demos_to_use_in_prompt)
        dict_responses['predicted_concepts'].append(concepts)

    # Convert to DataFrame
    df = pd.DataFrame(dict_responses)
    
    # Save results
    if model_name == "MMed":
        if split != None:
            file_path = f"results/label_prediction/{dataset}_split_{split}_{ckpt[ckpt.find('/')+1:]}_Explicd_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_n_demos_{n_demos}.csv"
        else:
            file_path = f"results/label_prediction/{dataset}_{ckpt[ckpt.find('/')+1:]}_Explicd_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_n_demos_{n_demos}.csv"
    elif model_name in ["Mistral", "GPT"]:
        if split != None:
            file_path = f"results/label_prediction/{dataset}_split_{split}_{model_name}_Explicd_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_n_demos_{n_demos}.csv"
        else:
            file_path = f"results/label_prediction/{dataset}_{model_name}_Explicd_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_n_demos_{n_demos}.csv"
    else:
        raise ValueError("Model name not recognized")
    
    # Extract the directory path from the file path
    dir_path = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


def classification(model_name:str, dataset: str, ckpt: str, split=None, ground_truth_concepts=False, raw_values=False, n_demos=0):
    """Calculate classification metrics from LLM predictions"""
    
    if model_name == "MMed":
        if split != None:
            df_responses = pd.read_csv(f"results/label_prediction/{dataset}_split_{split}_{ckpt[ckpt.find('/')+1:]}_Explicd_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_n_demos_{n_demos}.csv")
        else:
            df_responses = pd.read_csv(f"results/label_prediction/{dataset}_{ckpt[ckpt.find('/')+1:]}_Explicd_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_n_demos_{n_demos}.csv")
    elif model_name in ["Mistral", "GPT"]:
        if split != None:
            df_responses = pd.read_csv(f"results/label_prediction/{dataset}_split_{split}_{model_name}_Explicd_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_n_demos_{n_demos}.csv")
        else:
            df_responses = pd.read_csv(f"results/label_prediction/{dataset}_{model_name}_Explicd_raw_values_{raw_values}_gt_concepts_{ground_truth_concepts}_n_demos_{n_demos}.csv")
    else:
        raise ValueError("File not found!")

    if dataset == "PH2":
        PH2_TEST = pd.read_csv(f"data/PH2/splits/PH2_test_split_{split}.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(PH2_TEST.images.to_list())]
    elif dataset == "Derm7pt":
        D7_TEST = pd.read_csv("data/Derm7pt/splits/derm7pt_test.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(D7_TEST.images.to_list())]
    elif dataset == "HAM10000":
        HAM_TEST = pd.read_csv("data/HAM10000/splits/HAM10000_test.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(HAM_TEST.image_id.to_list())]

    mapping = {
        'nevus': 0,
        'melanoma': 1,
    }
    
    y_true = df_filtered.gt_response.map(mapping).to_list()
    y_pred = df_filtered.llm_response.map(mapping).to_list()
    
    # Get results
    results = calculate_metrics(y_true, y_pred)

    # Save results to JSON
    save_data_to_json(results, model=model_name, subdir='x_to_c_to_y', dataset=dataset, split=split, 
                     task=f"Explicd_gt_concepts_{ground_truth_concepts}_raw_values_{raw_values}_n_demos_{n_demos}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ExpLICD Self-Refine: Concept to Label Classification')
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate (PH2, Derm7pt, HAM10000)', default='PH2')
    parser.add_argument('--report_path', type=str, help='Path to custom report (optional)', default=None)
    parser.add_argument('--split', type=int, help='Split of the dataset if exists', default=None)
    parser.add_argument('--raw_values', action="store_true", help='Save raw concept scores')
    parser.add_argument('--ckpt', type=str, help='Name of the LLM checkpoint', default='Henrychur/MMed-Llama-3-8B')
    parser.add_argument('--llm', type=str, help='LLM for final diagnosis (MMed, Mistral, GPT)', default='MMed')
    parser.add_argument('--use_demos', action="store_true", help='Enable few-shot learning')
    parser.add_argument('--predict_for_train_set', action="store_true", help='Generate reports for training set')
    parser.add_argument('--n_demos', type=int, help='Number of demonstrations for few-shot', default=0)
    parser.add_argument('--gt_concepts', action="store_true", help='Use ground truth concepts')
    parser.add_argument('--data_path', type=str, 
                        default='/project/def-arashmoh/shahab33/Medsam/selff-ref/data',
                        help='Path to data directory')
    parser.add_argument('--generate_concepts', action="store_true", 
                        help='Generate concept predictions (x_to_c step)')
    args = parser.parse_args()

    seed_everything(seed=42)

    print("\n")
    print("#==============================================================================")
    print(f"# 🔬 ExpLICD Self-Refine Pipeline")
    print("#==============================================================================")
    print(f"# Status:    Running...")
    print(f"# Model:     ExpLICD (with self-refine)")
    print(f"# LLM:       {args.llm}")
    print(f"# Dataset:   {args.dataset}")
    print(f"# Split:     {args.split if args.split is not None else 'All'}")
    print(f"# n-shots:   {args.n_demos}")
    print(f"# Date:      {get_current_date()}")
    print("#==============================================================================")

    # Step 1: Generate concepts from images (x -> c)
    if args.generate_concepts:
        print("\n[STEP 1] Generating concept predictions from ExpLICD...")
        x_to_c(dataset=args.dataset, split=args.split, raw_values=args.raw_values, 
               predict_for_train_set=args.predict_for_train_set)
    
    # Step 2: Predict labels from concepts (c -> y)
    print("\n[STEP 2] Predicting final diagnosis from concepts using LLM...")
    c_to_y(model_name=args.llm, dataset=args.dataset, ckpt=args.ckpt, split=args.split, 
           raw_values=args.raw_values, report_path=args.report_path, use_demos=args.use_demos, 
           n_demos=args.n_demos, ground_truth_concepts=args.gt_concepts)
    
    # Step 3: Calculate classification metrics
    print("\n[STEP 3] Calculating classification metrics...")
    classification(model_name=args.llm, dataset=args.dataset, ckpt=args.ckpt, split=args.split, 
                   ground_truth_concepts=args.gt_concepts, raw_values=args.raw_values, n_demos=args.n_demos)

    print("\n")
    print("#==============================================================================")
    print(f"# Status:     ✅ Finished!")
    print(f"# Date:       {get_current_date()}")
    print("#==============================================================================")
    print("\n")
