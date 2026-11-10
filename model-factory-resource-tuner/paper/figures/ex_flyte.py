import flyte
from flyte.io import File
import pandas as pd

env = flyte.TaskEnvironment(name="pandas-join", image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"))


@env.task
async def join_orders_catalog(orders: File, catalog: File) -> dict:
    orders_path = await orders.download()
    catalog_path = await catalog.download()
    orders = pd.read_parquet(orders_path)
    catalog = pd.read_parquet(catalog_path)
    joined = orders.merge(catalog, on="sku", how="inner")
    joined["revenue"] = joined["qty"] * joined["price"]
    by_brand = joined.groupby("brand", observed=True)["revenue"].sum()
    return {"rows": int(len(joined)), "brands": int(len(by_brand))}


