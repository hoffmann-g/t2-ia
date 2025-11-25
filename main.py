import os
import pandas as pd
from openai import AsyncOpenAI
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

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLE_SIZE = 300

SYSTEM_PROMPT = """
You are an assistant specialized in classifying customer support intents for a banking application.
"""

PROMPT_WITH_EXAMPLES = """
You are an expert in classifying banking customer support messages.
Your task is to classify the message below into one of the intents from the provided list.

EXAMPLES:
1. Message: "I am still waiting on my card?"
   Intent: card_arrival
2. Message: "How do I know if my top up was successful?"
   Intent: top_up_failed
3. Message: "Why was I charged for making a purchase?"
   Intent: card_payment_fee_charged
4. Message: "I need to change my PIN number"
   Intent: change_pin
5. Message: "Someone stole my card, what do I do?"
   Intent: compromised_card

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


def get_unique_intents(df: pd.DataFrame) -> List[str]:
    """Extract list of unique intents from dataset."""
    return sorted(df['category'].dropna().unique().tolist())


async def classify_with_examples(message: str, intents: List[str]) -> str:
    """
    Classify a message using few-shot prompt (with examples).
    """
    intents_list = "\n".join([f"- {i}" for i in intents])
    prompt = PROMPT_WITH_EXAMPLES.format(
        intents_list=intents_list,
        message=message
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
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
            model="gpt-4o-mini",
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
    total_messages: int
) -> Tuple[str, str, str]:
    """
    Process a message by classifying with both methods in parallel.
    Returns: (actual_intent, prediction_with_examples, prediction_without_examples)
    """
    logger.info(f"[{idx + 1}/{total_messages}] Processing...")

    pred_with_examples, pred_without_examples = await asyncio.gather(
        classify_with_examples(message, unique_intents),
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
    logger.info("Dataset: Banking77 (Intent Classification)")
    logger.info("=" * 60 + "\n")

    logger.info("Loading dataset...")
    df = load_data("data/banking77_test.csv")
    logger.info(f"Test set size: {len(df)} messages\n")

    unique_intents = get_unique_intents(df)
    logger.info(f"Number of intents: {len(unique_intents)}")
    logger.info("Sample intents:")
    for i, intent in enumerate(unique_intents[:5], 1):
        logger.info(f"  {i}. {intent}")
    logger.info(f"  ... and {len(unique_intents) - 5} more\n")

    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        logger.info(f"Using random sample of {SAMPLE_SIZE} messages\n")

    total_messages = len(df)
    logger.info(f"Processing {total_messages} messages...\n")

    tasks = []
    for _, row in df.iterrows():
        task = process_message(
            idx=len(tasks),
            message=row['text'],
            actual_intent=row['category'],
            unique_intents=unique_intents,
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


def main():
    """Entry point that runs the async function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
