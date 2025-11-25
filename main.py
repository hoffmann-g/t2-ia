import os
import warnings
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from openai import AsyncOpenAI

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
from dotenv import load_dotenv
from typing import List, Tuple
import asyncio
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Ollama (local)
# MODEL = "gemma2:2b"
# client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# OpenAI
MODEL = "gpt-4o-mini"
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLES_PER_CLASS = 5
NUM_CLASSES = 20

SYSTEM_PROMPT = """
You are an assistant specialized in classifying customer support intents for a banking application.
"""

PROMPT_WITH_EXAMPLES = """
You are an expert in classifying banking customer support messages.
Your task is to classify the message below into one of the intents from the provided list.

EXAMPLES:
{examples}

AVAILABLE INTENTS:
{intents_list}

MESSAGE TO CLASSIFY:
{message}

Respond ONLY with the exact intent name from the list that best matches the message.
Do not add explanations or additional text.
"""

PROMPT_WITHOUT_EXAMPLES = """
You are an expert in classifying banking customer support messages.
Your task is to classify the message below into one of the intents from the provided list.

AVAILABLE INTENTS:
{intents_list}

MESSAGE TO CLASSIFY:
{message}

Respond ONLY with the exact intent name from the list that best matches the message.
Do not add explanations or additional text.
"""


def load_data(file_path: str) -> pd.DataFrame:
    """Load the Banking77 dataset."""
    return pd.read_csv(file_path)


def get_unique_intents(df: pd.DataFrame, num_classes: int = None) -> List[str]:
    """Extract list of unique intents from dataset."""
    intents = sorted(df['category'].dropna().unique().tolist())
    if num_classes:
        return intents[:num_classes]
    return intents


def get_balanced_sample(df: pd.DataFrame, samples_per_class: int) -> pd.DataFrame:
    """Get a balanced sample with N samples per class."""
    return df.groupby('category').apply(
        lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=42)
    ).reset_index(drop=True)


def build_examples_from_train(train_df: pd.DataFrame) -> str:
    """Build examples string with 1 example per category from training data."""
    examples = []
    for i, (intent, group) in enumerate(train_df.groupby('category'), 1):
        sample_text = group.iloc[0]['text']
        examples.append(f'{i}. Message: "{sample_text}"\n   Intent: {intent}')
    return "\n".join(examples)


async def classify_with_examples(message: str, intents: List[str], examples: str) -> str:
    """
    Classify a message using few-shot prompt (with examples).
    """
    intents_list = "\n".join([f"- {i}" for i in intents])
    prompt = PROMPT_WITH_EXAMPLES.format(
        examples=examples,
        intents_list=intents_list,
        message=message
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in classification with examples: {e}")
        return "ERROR"


async def classify_without_examples(message: str, intents: List[str]) -> str:
    """
    Classify a message using zero-shot prompt (without examples).
    """
    intents_list = "\n".join([f"- {i}" for i in intents])
    prompt = PROMPT_WITHOUT_EXAMPLES.format(
        intents_list=intents_list,
        message=message
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in classification without examples: {e}")
        return "ERROR"


async def process_message(
    idx: int,
    message: str,
    actual_intent: str,
    unique_intents: List[str],
    examples: str,
    total_messages: int
) -> Tuple[str, str, str]:
    """
    Process a message by classifying with both methods in parallel.
    Returns: (actual_intent, prediction_with_examples, prediction_without_examples)
    """
    logger.info(f"[{idx + 1}/{total_messages}] Processing...")

    pred_with_examples, pred_without_examples = await asyncio.gather(
        classify_with_examples(message, unique_intents, examples),
        classify_without_examples(message, unique_intents)
    )

    logger.info(f"  few-shot: {pred_with_examples}")
    logger.info(f"  zero-shot: {pred_without_examples}")
    logger.info(f"  actual: {actual_intent}\n")

    return actual_intent, pred_with_examples, pred_without_examples


def calculate_accuracy(predictions: List[str], actual: List[str]) -> float:
    """Calculate accuracy of predictions."""
    if len(predictions) != len(actual):
        raise ValueError("Predictions and actual lists have different sizes")

    correct = sum(1 for p, a in zip(predictions, actual) if p == a)
    return correct / len(actual) if len(actual) > 0 else 0.0


async def main_async():
    """Main async function."""
    logger.info("=" * 60)
    logger.info("FEW-SHOT vs ZERO-SHOT LEARNING COMPARISON")
    logger.info(f"Model: {MODEL}")
    logger.info("Dataset: Banking77 (Intent Classification)")
    logger.info("=" * 60 + "\n")

    logger.info("Loading datasets...")
    df = load_data("data/banking77_test.csv")
    train_df = load_data("data/banking77_train.csv")
    logger.info(f"Test set size: {len(df)} messages")
    logger.info(f"Train set size: {len(train_df)} messages\n")

    unique_intents = get_unique_intents(df, NUM_CLASSES)

    df = df[df['category'].isin(unique_intents)]
    train_df = train_df[train_df['category'].isin(unique_intents)]

    examples = build_examples_from_train(train_df)
    logger.info(f"Number of classes: {len(unique_intents)}")
    logger.info(f"Few-shot examples: {len(unique_intents)} (1 per class)")
    logger.info("Sample intents:")
    for i, intent in enumerate(unique_intents[:5], 1):
        logger.info(f"  {i}. {intent}")
    logger.info(f"  ... and {len(unique_intents) - 5} more\n")

    df = get_balanced_sample(df, SAMPLES_PER_CLASS)
    total_messages = len(df)
    logger.info(f"Test samples: {total_messages} ({SAMPLES_PER_CLASS} per class)\n")

    tasks = []
    for _, row in df.iterrows():
        task = process_message(
            idx=len(tasks),
            message=row['text'],
            actual_intent=row['category'],
            unique_intents=unique_intents,
            examples=examples,
            total_messages=total_messages
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    actual_intents = [r[0] for r in results]
    results_with_examples = [r[1] for r in results]
    results_without_examples = [r[2] for r in results]

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60 + "\n")

    accuracy_with_examples = calculate_accuracy(results_with_examples, actual_intents)
    accuracy_without_examples = calculate_accuracy(results_without_examples, actual_intents)

    logger.info(f"Few-shot accuracy:  {accuracy_with_examples:.2%} ({int(accuracy_with_examples * total_messages)}/{total_messages})")
    logger.info(f"Zero-shot accuracy: {accuracy_without_examples:.2%} ({int(accuracy_without_examples * total_messages)}/{total_messages})")
    logger.info(f"Difference: {abs(accuracy_with_examples - accuracy_without_examples):.2%}\n")

    if accuracy_with_examples > accuracy_without_examples:
        logger.info(">> Few-shot (with examples) performed better")
    elif accuracy_without_examples > accuracy_with_examples:
        logger.info(">> Zero-shot (without examples) performed better")
    else:
        logger.info(">> Both methods had the same performance")

    calculate_metrics(actual_intents, results_with_examples, "Few-Shot")
    calculate_metrics(actual_intents, results_without_examples, "Zero-Shot")


def calculate_metrics(actual: List[str], predicted: List[str], method_name: str):
    """Calculate and display precision, recall, F1 for a method."""
    all_labels = sorted(set(actual) | set(predicted))

    precision = precision_score(actual, predicted, labels=all_labels, average='weighted', zero_division=0)
    recall = recall_score(actual, predicted, labels=all_labels, average='weighted', zero_division=0)
    f1 = f1_score(actual, predicted, labels=all_labels, average='weighted', zero_division=0)

    precision_macro = precision_score(actual, predicted, labels=all_labels, average='macro', zero_division=0)
    recall_macro = recall_score(actual, predicted, labels=all_labels, average='macro', zero_division=0)
    f1_macro = f1_score(actual, predicted, labels=all_labels, average='macro', zero_division=0)

    logger.info(f"\n{method_name} Metrics:")
    logger.info(f"  Precision (weighted): {precision:.4f}")
    logger.info(f"  Recall (weighted):    {recall:.4f}")
    logger.info(f"  F1-Score (weighted):  {f1:.4f}")
    logger.info(f"  Precision (macro):    {precision_macro:.4f}")
    logger.info(f"  Recall (macro):       {recall_macro:.4f}")
    logger.info(f"  F1-Score (macro):     {f1_macro:.4f}")

    # Analyze errors per class
    analyze_errors(actual, predicted, method_name)


def analyze_errors(actual: List[str], predicted: List[str], method_name: str, top_n: int = 5):
    """Analyze and display classes with most errors."""
    from collections import Counter

    # Count errors per class
    errors_per_class = Counter()
    confusion_pairs = Counter()

    for a, p in zip(actual, predicted):
        if a != p:
            errors_per_class[a] += 1
            confusion_pairs[(a, p)] += 1

    if not errors_per_class:
        logger.info(f"\n  {method_name} - No errors found!")
        return

    logger.info(f"\n  {method_name} - Classes with most errors:")
    for intent, count in errors_per_class.most_common(top_n):
        logger.info(f"    {intent}: {count} errors")

    logger.info(f"\n  {method_name} - Most common confusions (actual -> predicted):")
    for (actual_intent, pred_intent), count in confusion_pairs.most_common(top_n):
        logger.info(f"    {actual_intent} -> {pred_intent}: {count}x")


def main():
    """Entry point that runs the async function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
