import pandas as pd
import json
import os
import boto3


def join_orders_catalog(orders_path: str, catalog_path: str) -> dict:
    orders = pd.read_parquet(orders_path)
    catalog = pd.read_parquet(catalog_path)
    joined = orders.merge(catalog, on="sku", how="inner")
    joined["revenue"] = joined["qty"] * joined["price"]
    by_brand = joined.groupby("brand", observed=True)["revenue"].sum()
    return {"rows": int(len(joined)), "brands": int(len(by_brand))}


def fetch(uri: str) -> str:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    path = os.path.join("/data", os.path.basename(key))
    s3.download_file(bucket, key, path)
    return path


if __name__ == "__main__":
    s3 = boto3.client("s3")
    result = join_orders_catalog(fetch(os.environ["INPUT_ORDERS"]), fetch(os.environ["INPUT_CATALOG"]))
    os.makedirs("/data/out", exist_ok=True)
    with open("/data/out/result.json", "w") as f:
        json.dump(result, f)
    bucket, key = os.environ["OUTPUT_URI"].removeprefix("s3://").split("/", 1)
    s3.upload_file("/data/out/result.json", bucket, key + "result.json")


