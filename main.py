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

FIELDS_TO_USE = [
    'description',
    # 'landing_page',
    # 'ad_id',
]

SYSTEM_PROMPT = """
You are an assistant specialized in product categorization.
"""

PROMPT_WITH_EXAMPLES = """
You are an expert in categorizing product advertisements.
Your task is to categorize the advertisement below into one of the products from the provided list.

EXAMPLES:
1. Ad: "Red light therapy belt for muscle recovery and pain relief"
   Correct product: "AllevaRed Light Therapy Belt"
2. Ad: "Vibration equipment for deep massage and muscle relaxation"
   Correct product: "Vibration Massage Device"

AVAILABLE PRODUCTS:
{products_list}

AD TO CATEGORIZE:
{description}

Respond ONLY with the exact product name from the list that best matches the advertisement.
Do not add explanations or additional text.
"""

PROMPT_WITHOUT_EXAMPLES = """
You are an expert in categorizing product advertisements.
Your task is to categorize the advertisement below into one of the products from the provided list.

AVAILABLE PRODUCTS:
{products_list}

AD TO CATEGORIZE:
{description}

Respond ONLY with the exact product name from the list that best matches the advertisement. Do not add explanations or additional text.
"""


def load_data(file_path: str) -> pd.DataFrame:
    """Load the ads dataset."""
    return pd.read_csv(file_path)


def get_unique_products(df: pd.DataFrame) -> List[str]:
    """Extract list of unique products from dataset."""
    return df['matching_product_title'].dropna().unique().tolist()


def build_ad_text(row: pd.Series, fields: List[str]) -> str:
    """
    Build the ad text from selected fields.

    Args:
        row: DataFrame row containing ad data
        fields: List of field names to include

    Returns:
        Formatted text with selected fields
    """
    parts = []
    for field in fields:
        if field in row and pd.notna(row[field]):
            value = str(row[field]).strip()
            if value:
                field_name = field.replace('_', ' ').title()
                parts.append(f"{field_name}: {value}")

    return "\n".join(parts) if parts else "No data available"


async def categorize_with_examples(ad_text: str, products: List[str]) -> str:
    """
    Categorize an ad using few-shot prompt (with examples).
    """
    products_list = "\n".join([f"- {p}" for p in products])
    prompt = PROMPT_WITH_EXAMPLES.format(
        products_list=products_list,
        description=ad_text
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in categorization with examples: {e}")
        return "ERROR"


async def categorize_without_examples(ad_text: str, products: List[str]) -> str:
    """
    Categorize an ad using zero-shot prompt (without examples).
    """
    products_list = "\n".join([f"- {p}" for p in products])
    prompt = PROMPT_WITHOUT_EXAMPLES.format(
        products_list=products_list,
        description=ad_text
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in categorization without examples: {e}")
        return "ERROR"


async def process_ad(
    idx: int,
    ad_id: str,
    ad_text: str,
    actual_product: str,
    unique_products: List[str],
    total_ads: int
) -> Tuple[str, str, str]:
    """
    Process an ad by categorizing with both methods in parallel.
    Returns: (actual_product, prediction_with_examples, prediction_without_examples)
    """
    logger.info(f"[{idx + 1}/{total_ads}] {ad_id}")

    pred_with_examples, pred_without_examples = await asyncio.gather(
        categorize_with_examples(ad_text, unique_products),
        categorize_without_examples(ad_text, unique_products)
    )

    logger.info(f"  with examples: {pred_with_examples}")
    logger.info(f"  without examples: {pred_without_examples}")
    logger.info(f"  actual: {actual_product}\n")

    return actual_product, pred_with_examples, pred_without_examples


def calculate_accuracy(predictions: List[str], actual: List[str]) -> float:
    """Calculate accuracy of predictions."""
    if len(predictions) != len(actual):
        raise ValueError("Predictions and actual lists have different sizes")

    correct = sum(1 for p, a in zip(predictions, actual) if p == a)
    return correct / len(actual) if len(actual) > 0 else 0.0


async def main_async():
    """Main async function."""
    logger.info("Starting categorization analysis\n")

    logger.info("Loading dataset...")
    df = load_data("data/ads.csv")
    logger.info(f"Dataset: {len(df)} ads\n")

    unique_products = get_unique_products(df)
    logger.info(f"Unique products: {len(unique_products)}")
    for i, product in enumerate(unique_products[:10], 1):
        logger.info(f"  {i}. {product}")
    if len(unique_products) > 10:
        logger.info(f"  ... and {len(unique_products) - 10} more\n")
    else:
        logger.info("")

    logger.info(f"Using fields: {', '.join(FIELDS_TO_USE)}\n")

    total_ads = len(df)
    logger.info(f"Processing {total_ads} ads in parallel...\n")

    tasks = []
    for idx, row in df.iterrows():
        ad_text = build_ad_text(row, FIELDS_TO_USE)
        task = process_ad(
            idx=idx,
            ad_id=row['ad_id'],
            ad_text=ad_text,
            actual_product=row['matching_product_title'],
            unique_products=unique_products,
            total_ads=total_ads
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    actual_products = [r[0] for r in results]
    results_with_examples = [r[1] for r in results]
    results_without_examples = [r[2] for r in results]

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60 + "\n")

    accuracy_with_examples = calculate_accuracy(results_with_examples, actual_products)
    accuracy_without_examples = calculate_accuracy(results_without_examples, actual_products)

    logger.info(f"Accuracy with examples: {accuracy_with_examples:.2%} ({int(accuracy_with_examples * total_ads)}/{total_ads})")
    logger.info(f"Accuracy without examples: {accuracy_without_examples:.2%} ({int(accuracy_without_examples * total_ads)}/{total_ads})")
    logger.info(f"Difference: {abs(accuracy_with_examples - accuracy_without_examples):.2%}\n")

    if accuracy_with_examples > accuracy_without_examples:
        logger.info("Prompts WITH examples performed better")
    elif accuracy_without_examples > accuracy_with_examples:
        logger.info("Prompts WITHOUT examples performed better")
    else:
        logger.info("Both methods had the same performance")

    results_df = pd.DataFrame({
        'ad_id': df['ad_id'],
        'actual_product': actual_products,
        'predicted_with_examples': results_with_examples,
        'predicted_without_examples': results_without_examples,
        'correct_with_examples': [p == a for p, a in zip(results_with_examples, actual_products)],
        'correct_without_examples': [p == a for p, a in zip(results_without_examples, actual_products)]
    })

    output_file = "results.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")


def main():
    """Entry point that runs the async function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
