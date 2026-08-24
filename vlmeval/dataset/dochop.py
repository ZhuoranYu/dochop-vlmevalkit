import re
import math
import pandas as pd
import numpy as np
from pathlib import Path
from vlmeval.dataset.image_vqa import ImageVQADataset
from vlmeval.smp import load

# ==================================================================================
#                                  Helper Functions
# ==================================================================================
SHARED_INSTRUCTION = (
    "\n\nYour final answer should be a single pure value:"
    "\n- If the question asks for an entity, use its complete name exactly as shown in the chart"
    "\n- If the answer is a number: output integers as integers, decimals rounded to two places (unless the question specifies otherwise)"
    "\n- Otherwise, follow the question's instructions for the expected format"
    "\n\nDo not include units, currency symbols, or percentage signs in your final answer."
    "\n\nAt the end of your response, format the final answer in a separate sentence like this:\n"
    "The answer is: <your_final_answer>"
)

def _clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r'^\s*answer\s*:\s*', '', s, flags=re.I)
    return s.strip()

def _extract_number(text: str):
    if not text:
        return None
    clean_text = text.replace(',', '')
    numbers = re.findall(r'-?\d+(?:\.\d+)?', clean_text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except:
        return None

def _is_correct(gt_str, pred_str):
    if _clean_text(gt_str).lower() == _clean_text(pred_str).lower(): 
        return True
    gt_val = _extract_number(gt_str)
    pred_val = _extract_number(pred_str)
    if gt_val is not None and pred_val is not None:
        return math.isclose(gt_val, pred_val, rel_tol=0.01, abs_tol=1e-3)
    return False

def extract_answer(response: str) -> str:
    if not response: return ""
    
    # Non-greedy match to end of line only
    pattern = r"the answer is[:\s]*(.+?)$"
    matches = re.findall(pattern, str(response), re.IGNORECASE | re.MULTILINE)

    if matches:
        raw_answer = matches[-1].strip()  # take the last match
        if raw_answer.endswith('.'): 
            raw_answer = raw_answer[:-1]
        return raw_answer.replace('*', '').replace('`', '').strip()
    
    # Fallback
    lines = str(response).strip().split('\n')
    return lines[-1].strip() if lines else ""

# ==================================================================================
#                                  DocHop
# ==================================================================================

class DocHop(ImageVQADataset):
    TYPE = "VQA"
    DATASET_URL = {
        "DocHop": "https://huggingface.co/datasets/zhuoranyu336/dochop/resolve/main/DocHop.tsv",
    }
    DATASET_MD5 = {
        "DocHop": "285a29d7b44e3113625380a39d48dc69",
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        intro = (
            "\nThe image provided is a document page containing both text and chart(s). "
            "Please read the text within the image and analyze the chart(s) to answer the question. "
        )
        full_instruction = intro + SHARED_INSTRUCTION
        if 'value' in msgs[-1]:
            msgs[-1]['value'] += full_instruction
        return msgs

    def evaluate_heuristic(self, eval_file, **judge_kwargs):
        print(f"[{self.__class__.__name__}] Evaluating {eval_file} ...")
        df = load(eval_file)
        
        # 1. Robust Task Column Detection
        # Check for common column names for 'Task'
        task_col = 'default'
        for cand in ['task', 'orig_task', 'category']:
            if cand in df.columns:
                task_col = cand
                break
        
        # Ensure columns exist, fill NaNs to prevent groupby errors
        if task_col not in df.columns: 
            df[task_col] = 'default'
        
        # Handle Depth and Chart Num columns if they exist, otherwise mark Unknown
        df['depth'] = df.get('depth', 'Unknown')
        df['chart_num'] = df.get('chart_num', 'Unknown')

        results = []
        for idx, row in df.iterrows():
            gt = str(row.get('answer', ''))
            pred = str(row.get('prediction', ''))
            
            clean_pred = extract_answer(pred) 
            correct = _is_correct(gt, clean_pred)
            
            results.append({
                'index': row.get('index', idx),
                'example_id': row.get('example_id', ''),
                'task': row[task_col],        # Normalized Task Column
                'depth': row['depth'],        
                'chart_num': row['chart_num'], 
                'question': row.get('question', ''),
                'answer': gt,
                'prediction': pred,
                'extracted': clean_pred,
                'correct': 1 if correct else 0
            })

        res_df = pd.DataFrame(results)
        
        # 2. Calculate Accuracies
        # Overall
        overall_acc = res_df['correct'].mean() * 100.0
        
        # Groupby metrics (Count and Mean)
        # Using agg to get count and mean in one go, avoiding re-calculation issues
        task_stats = res_df.groupby('task')['correct'].agg(['mean', 'count'])
        depth_stats = res_df.groupby('depth')['correct'].agg(['mean', 'count'])
        chart_stats = res_df.groupby('chart_num')['correct'].agg(['mean', 'count'])
        
        # 3. Build Summary DataFrame (CSV)
        stats_rows = []
        
        # [A] Overall
        stats_rows.append({
            'Dimension': 'Overall', 
            'Category': 'Total', 
            'Score': overall_acc, 
            'Count': len(res_df)
        })
        
        # [B] Per Task (The requested fix)
        for cat, row in task_stats.iterrows():
            stats_rows.append({
                'Dimension': 'Task', 
                'Category': cat, 
                'Score': row['mean'] * 100.0, 
                'Count': row['count']
            })
            
        # [C] Per Depth (Sorted safely)
        # Try to sort numerically if possible, else strictly string sort
        try:
            sorted_indices = sorted(depth_stats.index, key=lambda x: int(x) if str(x).isdigit() else 999)
        except:
            sorted_indices = sorted(depth_stats.index, key=lambda x: str(x))

        for cat in sorted_indices:
            row = depth_stats.loc[cat]
            stats_rows.append({
                'Dimension': 'Depth', 
                'Category': cat, 
                'Score': row['mean'] * 100.0, 
                'Count': row['count']
            })
            
        # [D] Per Chart Num (Sorted safely)
        try:
            sorted_indices = sorted(chart_stats.index, key=lambda x: int(x) if str(x).isdigit() else 999)
        except:
            sorted_indices = sorted(chart_stats.index, key=lambda x: str(x))

        for cat in sorted_indices:
            row = chart_stats.loc[cat]
            stats_rows.append({
                'Dimension': 'Chart Num', 
                'Category': cat, 
                'Score': row['mean'] * 100.0, 
                'Count': row['count']
            })

        summary_df = pd.DataFrame(stats_rows)

        # 4. Save Files
        eval_path = Path(eval_file)
        
        # Save Details
        res_df.to_excel(eval_path.with_name(f"{eval_path.stem}_details.xlsx"), index=False)
        
        # Save Summary CSV (This now definitely contains Task, Depth, Chart Num)
        csv_path = eval_path.with_name(f"{eval_path.stem}_acc.csv")
        summary_df.to_csv(csv_path, index=False)

        # 5. Console Output
        print(f"\n{'='*60}")
        print(f"📊 Evaluation Report Saved to: {csv_path.name}")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        
        print(f"\n🔹 By Task:")
        print(task_stats['mean'].mul(100).map('{:.2f}%'.format).to_string())
        
        print(f"\n🔹 By Depth:")
        print(depth_stats['mean'].mul(100).map('{:.2f}%'.format).to_string())
        
        print(f"\n🔹 By Chart Num:")
        print(chart_stats['mean'].mul(100).map('{:.2f}%'.format).to_string())
        print(f"{'='*60}\n")

        # Return simple report for VLMEvalKit compatibility
        return pd.DataFrame([{'Metric': 'Accuracy', 'Score': overall_acc}])