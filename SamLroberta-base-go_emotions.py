import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import ast
from scipy.stats import chi2_contingency, fisher_exact, norm
from statsmodels.stats.multitest import multipletests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm
import os

# load data
PTH = '/......'

# Enable GPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")


# Load the pre-trained GoEmotions model
classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    truncation=True,
    device=device,
    max_length=512,
    top_k=None  # Returns scores for all 28 emotion labels
)

df = pd.read_csv(os.path.join(PTH,'AIVI_HI.csv'),low_memory=False)

df = df[~df['posts.comments.text'].isna()]

# Extract comments
comments = df["posts.comments.text"].fillna("").tolist()

# Function to process emotions from predictions
def extract_emotions(predictions, threshold=0.3):
    emotions = [pred["label"] for pred in predictions if pred["score"] > threshold]
    scores = [pred["score"] for pred in predictions if pred["score"] > threshold]
    return emotions, scores


# Process in batches with progress bar
batch_size = 32
all_emotions = []
all_scores = []


for i in tqdm(range(0, len(comments), batch_size), desc="Processing batches"):
    batch = comments[i:i+batch_size]

    # Batch prediction
    predictions_batch = classifier(batch)

    # Extract emotions for each comment in batch
    for predictions in predictions_batch:
        emotions, scores = extract_emotions(predictions, threshold=0.3)
        all_emotions.append(emotions)
        all_scores.append(scores)

    # Save checkpoint every 20,000 comments
    if (i + batch_size) % 20000 == 0:
        checkpoint_df = df.iloc[:len(all_emotions)].copy()
        checkpoint_df["emotions"] = all_emotions
        checkpoint_df["emotion_scores"] = all_scores
        checkpoint_df.to_csv(f"checkpoint_{i}.csv", index=False)
        print(f"Checkpoint saved at {i} comments")

# Add results to dataframe
df["emotions"] = all_emotions
df["emotion_scores"] = all_scores

# Save final results
df.to_csv(os.path.join(PTH,"comments_with_emotions.csv"), index=False)
print("Processing complete!")
print(df.head())

df = pd.read_csv(os.path.join(PTH,'comments_with_emotions.csv'), low_memory=False)


# =========================================================
# Statisticak testing.
# ---------------------------------------------------------
# df must contain:
#   - user_type : values 'AI' and 'HUMAN'
#   - emotions  : list of emotions OR string representation of list
# =========================================================

# ---------------------------
# PREPARE DATA
# ---------------------------
df = df.copy()

def parse_emotions(x):
    if isinstance(x, list):
        return list(set(x))
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return list(set(parsed))
            return []
        except Exception:
            return []
    return []

df['emotions'] = df['emotions'].apply(parse_emotions)
df = df[df['user_type'].isin(['AI', 'HUMAN'])].copy()

n_ai = (df['user_type'] == 'AI').sum()
n_human = (df['user_type'] == 'HUMAN').sum()

print(f"Total AI comments: {n_ai}")
print(f"Total HUMAN comments: {n_human}")

# All unique emotions
all_emotions = sorted(set(e for lst in df['emotions'] for e in lst))

# ---------------------------
# HELP FUNCTIONS
# ---------------------------
def safe_div(a, b):
    return np.nan if b == 0 else a / b

def cohens_h(p1, p2):
    """Effect size for difference between two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def interpret_h(h):
    ah = abs(h)
    if ah < 0.10:
        return "trivial"
    elif ah < 0.20:
        return "very small"
    elif ah < 0.50:
        return "small"
    elif ah < 0.80:
        return "medium"
    else:
        return "large"

def interpret_phi(phi):
    ap = abs(phi)
    if ap < 0.10:
        return "negligible"
    elif ap < 0.30:
        return "small"
    elif ap < 0.50:
        return "medium"
    else:
        return "large"

def compute_or_and_ci(a, b, c, d, alpha=0.05, correction=True):
    """
    2x2 table:
              Present  Absent
    AI          a        b
    HUMAN       c        d

    Returns:
        odds_ratio, ci_low, ci_high
    """
    # Haldane-Anscombe correction if any cell is zero
    if correction and min(a, b, c, d) == 0:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = a, b, c, d

    odds_ratio = (a_ * d_) / (b_ * c_)

    # Wald CI on log(OR)
    se_log_or = np.sqrt(1/a_ + 1/b_ + 1/c_ + 1/d_)
    z = norm.ppf(1 - alpha / 2)
    log_or = np.log(odds_ratio)
    ci_low = np.exp(log_or - z * se_log_or)
    ci_high = np.exp(log_or + z * se_log_or)

    return odds_ratio, ci_low, ci_high

def risk_ratio_and_ci(a, b, c, d, alpha=0.05, correction=True):
    """
    Risk ratio for:
      risk_AI    = a / (a+b)
      risk_HUMAN = c / (c+d)
    """
    if correction and min(a, b, c, d) == 0:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = a, b, c, d

    risk_ai = a_ / (a_ + b_)
    risk_human = c_ / (c_ + d_)
    rr = risk_ai / risk_human

    se_log_rr = np.sqrt((1/a_) - (1/(a_ + b_)) + (1/c_) - (1/(c_ + d_)))
    z = norm.ppf(1 - alpha / 2)
    log_rr = np.log(rr)
    ci_low = np.exp(log_rr - z * se_log_rr)
    ci_high = np.exp(log_rr + z * se_log_rr)

    return rr, ci_low, ci_high

# ---------------------------
# MAIN ANALYSIS
# ---------------------------
results = []

for emotion in all_emotions:
    # Presence of emotion at comment level
    present_mask = df['emotions'].apply(lambda x: emotion in x)

    ai_present = ((df['user_type'] == 'AI') & present_mask).sum()
    human_present = ((df['user_type'] == 'HUMAN') & present_mask).sum()

    ai_absent = n_ai - ai_present
    human_absent = n_human - human_present

    # 2x2 table
    #            Present   Absent
    # AI         a         b
    # HUMAN      c         d
    a, b, c, d = ai_present, ai_absent, human_present, human_absent

    table = np.array([[a, b],
                      [c, d]])

    # Expected counts
    chi2, p_chi2, dof, expected = chi2_contingency(table, correction=False)

    # Use Fisher when expected counts are very small
    use_fisher = (expected < 5).any()
    if use_fisher:
        oddsratio_fisher, p_value = fisher_exact(table, alternative='two-sided')
        test_used = 'fisher'
    else:
        p_value = p_chi2
        test_used = 'chi2'

    # Phi = Cramér's V for 2x2
    n_total = table.sum()
    phi = np.sqrt(chi2 / n_total)

    # Proportions
    p_ai = a / n_ai
    p_human = c / n_human

    ai_pct = p_ai * 100
    human_pct = p_human * 100
    diff_pct_points = ai_pct - human_pct

    # Cohen's h
    h = cohens_h(p_ai, p_human)

    # Odds Ratio + CI
    odds_ratio, or_ci_low, or_ci_high = compute_or_and_ci(a, b, c, d)

    # Risk Ratio + CI
    rr, rr_ci_low, rr_ci_high = risk_ratio_and_ci(a, b, c, d)

    # Relative increase/decrease in prevalence
    rel_change_pct = ((p_ai - p_human) / p_human * 100) if p_human > 0 else np.nan

    results.append({
        'Emotion': emotion,
        'AI_count': a,
        'AI_absent': b,
        'HUMAN_count': c,
        'HUMAN_absent': d,
        'AI_pct': ai_pct,
        'HUMAN_pct': human_pct,
        'Difference_pct_points': diff_pct_points,
        'Relative_change_pct': rel_change_pct,
        'chi2': chi2,
        'p_value': p_value,
        'test_used': test_used,
        'phi': phi,
        'phi_interpretation': interpret_phi(phi),
        'cohens_h': h,
        'abs_cohens_h': abs(h),
        'h_interpretation': interpret_h(h),
        'odds_ratio': odds_ratio,
        'or_ci_low': or_ci_low,
        'or_ci_high': or_ci_high,
        'risk_ratio': rr,
        'rr_ci_low': rr_ci_low,
        'rr_ci_high': rr_ci_high
    })

results_df = pd.DataFrame(results)

# ---------------------------
# MULTIPLE TESTING
# ---------------------------
alpha = 0.05

reject_bonf, p_bonf, _, _ = multipletests(results_df['p_value'], alpha=alpha, method='bonferroni')
reject_fdr, p_fdr, _, _ = multipletests(results_df['p_value'], alpha=alpha, method='fdr_bh')

results_df['p_bonferroni'] = p_bonf
results_df['significant_bonferroni'] = reject_bonf
results_df['p_fdr'] = p_fdr
results_df['significant_fdr'] = reject_fdr

# ---------------------------
# SORTING FOR REPORTING
# ---------------------------
results_df = results_df.sort_values(
    by=['abs_cohens_h', 'Difference_pct_points'],
    ascending=[False, False]
).reset_index(drop=True)

# ---------------------------
# SAVE OUTPUTS
# ---------------------------
results_df.to_csv('emotion_effect_sizes_full.csv', index=False)

manuscript_cols = [
    'Emotion',
    'AI_count',
    'HUMAN_count',
    'AI_pct',
    'HUMAN_pct',
    'Difference_pct_points',
    'Relative_change_pct',
    'p_value',
    'p_bonferroni',
    'p_fdr',
    'phi',
    'cohens_h',
    'h_interpretation',
    'odds_ratio',
    'or_ci_low',
    'or_ci_high',
    'risk_ratio',
    'rr_ci_low',
    'rr_ci_high'
]
results_df[manuscript_cols].to_csv('emotion_effect_sizes_manuscript.csv', index=False)

# ---------------------------
# DISPLAY RESULTS
# ---------------------------
print("\n" + "=" * 90)
print("TOP 12 EMOTIONS BY ABSOLUTE COHEN'S h")
print("=" * 90)
top12 = results_df.head(12)

display_cols = [
    'Emotion',
    'AI_pct',
    'HUMAN_pct',
    'Difference_pct_points',
    'phi',
    'cohens_h',
    'h_interpretation',
    'odds_ratio',
    'or_ci_low',
    'or_ci_high',
    'p_bonferroni',
    'significant_bonferroni'
]
print(top12[display_cols].to_string(index=False))

print("\n" + "=" * 90)
print("SIGNIFICANT RESULTS AFTER BONFERRONI")
print("=" * 90)
sig = results_df[results_df['significant_bonferroni']].copy()
print(f"Number significant after Bonferroni: {len(sig)} / {len(results_df)}\n")

for _, row in sig.iterrows():
    direction = "AI > HUMAN" if row['Difference_pct_points'] > 0 else "HUMAN > AI"
    print(
        f"{row['Emotion']:15s} | "
        f"AI={row['AI_pct']:.2f}% | HUMAN={row['HUMAN_pct']:.2f}% | "
        f"Δ={row['Difference_pct_points']:+.2f} pp | "
        f"h={row['cohens_h']:+.3f} ({row['h_interpretation']}) | "
        f"OR={row['odds_ratio']:.3f} "
        f"[{row['or_ci_low']:.3f}, {row['or_ci_high']:.3f}] | "
        f"{direction}"
    )

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)
print("1. emotion_effect_sizes_full.csv")
print("2. emotion_effect_sizes_manuscript.csv")



# ------------ Load the manuscript data ---------------------
df = pd.read_csv("emotion_effect_sizes_manuscript.csv")

# 3. Sort strictly by absolute Cohen's h (Largest effect at the top)
plot_df = (
    df.assign(abs_h=df["cohens_h"].abs())
      .sort_values("abs_h", ascending=True) # Ascending=True because barh plots from bottom up
      .tail(12)
      .copy()
)

plot_df["label"] = plot_df["Emotion"].str.capitalize()

# 12 Top Emotion Differences Between AIVI and HI Directed Comments
colors = ["#FF6B6B" if x > 0 else "#4ECDC4" for x in plot_df["Difference_pct_points"]]

# figure size
fig, ax = plt.subplots(figsize=(12, 10))

bars = ax.barh(
    plot_df["label"],
    plot_df["Difference_pct_points"],
    color=colors,
    edgecolor='black',
    linewidth=0.5
)

# Add zero line
ax.axvline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.8)

# Visual expllegened for direction (Text above the axis)
ax.text(3, len(plot_df), "AIVI > HI", color="#FF6B6B", fontsize=16, fontweight='bold', ha='center', va='bottom')
ax.text(-3, len(plot_df), "HI > AIVI", color="#4ECDC4", fontsize=16, fontweight='bold', ha='center', va='bottom')

# Annotate with Cohen's h
for bar, h in zip(bars, plot_df["cohens_h"]):
    x = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    txt = f"h={h:+.3f}"
    # Increased padding to 1.2/-1.2 to avoid crowding
    pad = 1.2 if x >= 0 else -1.2
    ax.text(x + pad, y, txt, va="center", ha="left" if x >= 0 else "right", fontsize=15, fontweight='bold')

# title
ax.set_title("Top 12 Emotion Differences Between AIVI- and HI-Directed Comments", 
             fontsize=20, fontweight='bold', pad=45)

# X-label and Ticks
ax.set_xlabel("Difference in Percentage Points (%AIVI - %HI)", fontsize=18, labelpad=15)
ax.tick_params(axis='both', labelsize=16)

# x-limits
ax.set_xlim(-15, 15)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend at bottom
legend_elements = [Patch(facecolor='#FF6B6B', label='AIVI > HI'),
                   Patch(facecolor='#4ECDC4', label='HI > AIVI')]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=16, ncol=2)

# Adjust layout
plt.subplots_adjust(left=0.25, bottom=0.2, top=0.88)
plt.savefig("figure4_emotions_final_refined.pdf", dpi=300, bbox_inches="tight")
plt.show()
