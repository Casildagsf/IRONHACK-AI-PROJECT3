import pandas as pd
import torch


# ==========================================================
# Data Preparation
# ==========================================================

def aggregate_reviews(reviews):
    """
    Aggregate all review texts by product category.
    """

    category_reviews = (
        reviews
        .groupby("cluster_name")["reviews.text"]
        .apply(list)
        .reset_index()
    )

    return category_reviews


def get_top_products(products, reviews, top_n=3):
    """
    Identify the top N products in each category.
    """

    rankable_products = products[
        products["rankable"]
    ].copy()

    rankable_products = rankable_products.sort_values(
        by=[
            "cluster_name",
            "mean_rating",
            "n_reviews"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )

    top_products = (
        rankable_products
        .groupby("cluster_name")
        .head(top_n)
        .reset_index(drop=True)
    )

    product_lookup = (
        reviews[
            ["asins", "name"]
        ]
        .drop_duplicates(subset="asins")
    )

    top_products = top_products.merge(
        product_lookup,
        on="asins",
        how="left"
    )

    top_products["name"] = (
        top_products["name"]
        .fillna(top_products["brand"])
    )

    return top_products


def get_lowest_products(products, reviews):
    """
    Identify the lowest-rated product in each category.
    """

    rankable_products = products[
        products["rankable"]
    ].copy()

    rankable_products = rankable_products.sort_values(
        by=[
            "cluster_name",
            "mean_rating",
            "n_reviews"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )

    lowest_products = (
        rankable_products
        .groupby("cluster_name")
        .head(1)
        .reset_index(drop=True)
    )

    product_lookup = (
        reviews[
            ["asins", "name"]
        ]
        .drop_duplicates(subset="asins")
    )

    lowest_products = lowest_products.merge(
        product_lookup,
        on="asins",
        how="left"
    )

    lowest_products["name"] = (
        lowest_products["name"]
        .fillna(lowest_products["brand"])
    )

    return lowest_products


# ==========================================================
# Review Processing
# ==========================================================

def get_review_examples(
    reviews,
    sentiment,
    n_examples=5
):
    """
    Return representative review examples for a given sentiment.
    """

    sentiment_reviews = reviews[
        reviews["sentiment"] == sentiment
    ].copy()

    examples = (
        sentiment_reviews
        .groupby("cluster_name")["reviews.text"]
        .apply(lambda x: x.head(n_examples).tolist())
        .reset_index()
    )

    return examples


# ==========================================================
# Prompt Engineering
# ==========================================================

def build_prompt(
    category,
    top_products,
    lowest_products,
    positive_examples,
    negative_examples
):
    """
    Build the prompt for one product category.
    """

    top = top_products.loc[
        top_products["cluster_name"] == category,
        "name"
    ].tolist()

    worst = lowest_products.loc[
        lowest_products["cluster_name"] == category,
        "name"
    ].iloc[0]

    positive = positive_examples.loc[
        positive_examples["cluster_name"] == category,
        "reviews.text"
    ].iloc[0]

    negative = negative_examples.loc[
        negative_examples["cluster_name"] == category,
        "reviews.text"
    ].iloc[0]

    prompt = f"""
You are an expert technology reviewer.

Write a consumer buying guide (150–200 words) for the following product category.

Category:
{category}

Top Products:
{chr(10).join("- " + p for p in top)}

Lowest Rated Product:
- {worst}

Positive Customer Feedback:
{chr(10).join("- " + r for r in positive)}

Negative Customer Feedback:
{chr(10).join("- " + r for r in negative)}

Include:

- The three best products.
- Their main strengths.
- The most common complaints.
- Which product should be avoided and why.

Write in a neutral and informative tone.
"""

    return prompt


# ==========================================================
# Text Generation
# ==========================================================

def generate_summary(
    prompt,
    tokenizer,
    model,
    device
):
    """
    Generate a summary using FLAN-T5.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=220,
        temperature=0.7,
        do_sample=True
    )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary